#include "mcc.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace {

// Numeric validation

constexpr double epsilon = 16 * std::numeric_limits<double>::epsilon();

bool close(Time a, Time b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <=
               epsilon * std::max({1.0, std::abs(a), std::abs(b)});
}

// Deadlines and non-overlap constraints are hard. Equality tolerances are used
// only when independently checking recomputed floating-point equations.
bool exceeds(Time a, Time b) { return a > b; }

Time add(Time a, Time b) {
    const Time result = a + b;
    if (!std::isfinite(result)) {
        throw std::overflow_error("Timing overflow");
    }
    return result;
}

void check_deadline(Time deadline) {
    if (!std::isfinite(deadline) || deadline < 0) {
        throw std::invalid_argument("Deadline must be finite and nonnegative");
    }
}

void check_energy(std::size_t k, const std::vector<double>& powers, double sending) {
    if (powers.size() != k || !std::isfinite(sending) || sending < 0) {
        throw std::invalid_argument(
            "Energy model must have K finite nonnegative powers and RF power");
    }
    for (const double power : powers) {
        if (!std::isfinite(power) || power < 0) {
            throw std::invalid_argument("Core powers must be finite and nonnegative");
        }
    }
}

double energy_value(const Task& task, const std::vector<double>& powers,
                    double sending_power) {
    if (task.core_execution_times.size() != powers.size() || task.assignment < 0 ||
        static_cast<std::size_t>(task.assignment) > powers.size()) {
        throw std::invalid_argument("Energy requires valid assignments and K powers");
    }
    const std::size_t assignment = static_cast<std::size_t>(task.assignment);
    const double energy = assignment == powers.size()
                              ? sending_power * task.cloud_execution_times[0]
                              : powers[assignment] * task.core_execution_times[assignment];
    if (!std::isfinite(energy) || energy < 0) {
        throw std::overflow_error("Invalid/overflowing task energy");
    }
    return energy;
}

// Compact graph representation

class Adjacency {
public:
    using Iterator = std::vector<std::size_t>::const_iterator;

    struct Range {
        Iterator first;
        Iterator last;

        Iterator begin() const { return first; }
        Iterator end() const { return last; }
        std::size_t size() const { return static_cast<std::size_t>(last - first); }
    };

    Adjacency() = default;

    explicit Adjacency(const std::vector<std::size_t>& degrees)
        : offsets_(degrees.size() + 1, 0) {
        for (std::size_t row = 0; row < degrees.size(); ++row) {
            offsets_[row + 1] = offsets_[row] + degrees[row];
        }
        neighbors_.resize(offsets_.back());
    }

    void set(std::size_t row, std::size_t position, std::size_t neighbor) {
        neighbors_[offsets_[row] + position] = neighbor;
    }

    Range operator[](std::size_t row) const {
        return {neighbors_.cbegin() + static_cast<std::ptrdiff_t>(offsets_[row]),
                neighbors_.cbegin() + static_cast<std::ptrdiff_t>(offsets_[row + 1])};
    }

private:
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbors_;
};

// Built once per scheduling pass. IDs use expected O(1) hash lookup while
// predecessor and successor lists use compact CSR storage for cache-local scans.
struct Graph {
    std::size_t k;
    std::unordered_map<int, std::size_t> index;
    Adjacency pred;
    Adjacency succ;
    std::vector<std::size_t> topological;

    explicit Graph(const std::vector<Task>& tasks, bool check_durations = true)
        : k(tasks.empty() ? 0 : tasks[0].core_execution_times.size()) {
        const std::size_t n = tasks.size();
        if (n == 0 || k == 0 ||
            n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            k > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("A graph needs at least one task and one core");
        }

        index.max_load_factor(0.7F);
        index.reserve(n);
        std::vector<std::size_t> predecessor_degrees(n);
        std::vector<std::size_t> successor_degrees(n);
        std::size_t edge_count = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& task = tasks[i];
            if (!index.emplace(task.id, i).second) {
                throw std::invalid_argument("Duplicate task ID");
            }
            if (task.core_execution_times.size() != k) {
                throw std::invalid_argument("Inconsistent core count");
            }
            if (check_durations) {
                for (const Time duration : task.core_execution_times) {
                    if (!std::isfinite(duration) || duration <= 0) {
                        throw std::invalid_argument(
                            "Local durations must be finite and positive");
                    }
                }
            }
            for (std::size_t phase = 0; phase < 3; ++phase) {
                const Time duration = task.cloud_execution_times[phase];
                if (!std::isfinite(duration) || duration < 0 ||
                    (phase < 2 && duration == 0)) {
                    throw std::invalid_argument(
                        "Upload/compute durations must be positive; receive may be zero");
                }
            }

            predecessor_degrees[i] = task.pred_tasks.size();
            successor_degrees[i] = task.succ_tasks.size();
            edge_count += predecessor_degrees[i];
        }

        pred = Adjacency(predecessor_degrees);
        succ = Adjacency(successor_degrees);

        std::unordered_set<std::uint64_t> unmatched_edges;
        unmatched_edges.max_load_factor(0.7F);
        unmatched_edges.reserve(edge_count);
        const auto edge_key = [n](std::size_t from, std::size_t to) {
            return std::uint64_t(from) * n + to;
        };

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t position = 0;
            for (const int id : tasks[i].pred_tasks) {
                const auto found = index.find(id);
                if (found == index.end() || found->second == i) {
                    throw std::invalid_argument("Unknown/self predecessor");
                }
                const std::size_t predecessor = found->second;
                if (!unmatched_edges.insert(edge_key(predecessor, i)).second) {
                    throw std::invalid_argument("Duplicate edge");
                }
                pred.set(i, position++, predecessor);
            }
        }

        for (std::size_t i = 0; i < n; ++i) {
            std::size_t position = 0;
            for (const int id : tasks[i].succ_tasks) {
                const auto found = index.find(id);
                if (found == index.end() ||
                    unmatched_edges.erase(edge_key(i, found->second)) != 1) {
                    throw std::invalid_argument("Inconsistent or duplicate successor edge");
                }
                succ.set(i, position++, found->second);
            }
        }
        if (!unmatched_edges.empty()) {
            throw std::invalid_argument("Missing reciprocal successor edge");
        }

        topological.reserve(n);
        std::vector<std::size_t> remaining(n);
        for (std::size_t i = 0; i < n; ++i) {
            remaining[i] = pred[i].size();
            if (remaining[i] == 0) {
                topological.push_back(i);
            }
        }
        for (std::size_t position = 0; position < topological.size(); ++position) {
            for (const std::size_t child : succ[topological[position]]) {
                if (--remaining[child] == 0) {
                    topological.push_back(child);
                }
            }
        }
        if (topological.size() != n) {
            throw std::invalid_argument("Task graph contains a cycle");
        }
    }
};

// Timing and resource calendars

void reset_timing(Task& task) {
    task.FT_l = task.FT_ws = task.FT_c = task.FT_wr = 0;
    task.RT_l = task.RT_ws = task.RT_c = task.RT_wr = 0;
    task.execution_finish_time = 0;
    task.execution_start_time = 0;
    task.is_scheduled = SchedulingState::UNSCHEDULED;
}

template <typename Predecessors>
void dependency_times(Task& task, const std::vector<Task>& tasks,
                      const Predecessors& predecessors) {
    task.RT_l = task.RT_ws = 0;
    for (const std::size_t predecessor : predecessors) {
        task.RT_l = std::max(
            task.RT_l,
            std::max(tasks[predecessor].FT_l, tasks[predecessor].FT_wr));
        task.RT_ws = std::max(
            task.RT_ws,
            std::max(tasks[predecessor].FT_l, tasks[predecessor].FT_ws));
    }
}

void place_local(Task& task, std::size_t core, Time start) {
    if (!std::isfinite(task.core_execution_times[core]) ||
        task.core_execution_times[core] <= 0) {
        throw std::invalid_argument("Active local duration must be finite and positive");
    }
    task.assignment = static_cast<int>(core);
    task.is_core_task = true;
    task.execution_start_time = start;
    task.FT_l = add(start, task.core_execution_times[core]);
    task.execution_finish_time = task.FT_l;
}
struct CloudTiming {
    Time upload_finish;
    Time cloud_ready;
    Time cloud_finish;
    Time return_finish;
};

template <typename Predecessors>
CloudTiming calculate_cloud_timing(const Task& task, Time start,
                                   const std::vector<Task>& tasks,
                                   const Predecessors& predecessors) {
    CloudTiming timing;
    timing.upload_finish = add(start, task.cloud_execution_times[0]);
    timing.cloud_ready = timing.upload_finish;
    for (const auto predecessor : predecessors) {
        timing.cloud_ready = std::max(timing.cloud_ready, tasks[predecessor].FT_c);
    }
    timing.cloud_finish = add(timing.cloud_ready, task.cloud_execution_times[1]);
    timing.return_finish = add(timing.cloud_finish, task.cloud_execution_times[2]);
    return timing;
}

template <typename Predecessors>
void place_cloud(Task& task, std::size_t k, Time start,
                 const std::vector<Task>& tasks, const Predecessors& predecessors) {
    const CloudTiming timing = calculate_cloud_timing(task, start, tasks, predecessors);
    task.assignment = static_cast<int>(k);
    task.is_core_task = false;
    task.execution_start_time = start;
    task.FT_ws = timing.upload_finish;
    task.RT_c = timing.cloud_ready;
    task.FT_c = timing.cloud_finish;
    task.RT_wr = task.FT_c;
    task.FT_wr = timing.return_finish;
    task.execution_finish_time = task.FT_wr;
}

struct Interval {
    Time start;
    Time finish;
    int id;
};

Time earliest_slot(const std::vector<Interval>& calendar, Time ready, Time duration) {
    Time start = ready;
    for (const auto& interval : calendar) {
        if (add(start, duration) <= interval.start) {
            break;
        }
        start = std::max(start, interval.finish);
    }
    return start;
}

void insert_interval(std::vector<Interval>& calendar, Interval interval) {
    const auto position = std::lower_bound(
        calendar.begin(), calendar.end(), interval.start,
        [](const Interval& current, Time start) { return current.start < start; });
    calendar.insert(position, interval);
}

void primary_assignment_impl(std::vector<Task>& tasks, const Graph& graph,
                             bool cloud_enabled) {
    for (auto& task : tasks) {
        reset_timing(task);
        const Time remote = add(
            add(task.cloud_execution_times[0], task.cloud_execution_times[1]),
            task.cloud_execution_times[2]);
        const Time fastest_local = *std::min_element(task.core_execution_times.begin(),
                                                     task.core_execution_times.end());
        task.is_core_task = !(cloud_enabled && remote < fastest_local);
        task.assignment = task.is_core_task ? -1 : static_cast<int>(graph.k);
    }
}

void task_prioritizing_impl(std::vector<Task>& tasks, const Graph& graph) {
    for (auto position = graph.topological.rbegin();
         position != graph.topological.rend(); ++position) {
        auto& task = tasks[*position];
        const double weight = task.is_core_task
                                  ? std::accumulate(task.core_execution_times.begin(),
                                                    task.core_execution_times.end(), 0.0) /
                                        graph.k
                                  : std::accumulate(task.cloud_execution_times.begin(),
                                                    task.cloud_execution_times.end(), 0.0);
        double downstream = 0;
        for (const std::size_t child : graph.succ[*position]) {
            downstream = std::max(downstream, tasks[child].priority_score);
        }
        task.priority_score = add(weight, downstream);
    }
}

// Initial scheduling

Sequences select_units(std::vector<Task>& tasks, const Graph& graph, bool cloud_enabled,
                       bool fixed) {
    std::vector<std::vector<Interval>> calendars(graph.k + 1);
    std::vector<std::size_t> remaining(tasks.size());

    // Equal ranks: larger task ID first, an explicit deterministic convention.
    const auto lower_priority = [&](std::size_t left, std::size_t right) {
        if (tasks[left].priority_score != tasks[right].priority_score) {
            return tasks[left].priority_score < tasks[right].priority_score;
        }
        return tasks[left].id < tasks[right].id;
    };

    std::vector<std::size_t> heap_storage;
    heap_storage.reserve(tasks.size());
    std::priority_queue<std::size_t, std::vector<std::size_t>,
                        decltype(lower_priority)>
        ready(lower_priority, std::move(heap_storage));

    for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (!std::isfinite(tasks[i].priority_score)) {
            throw std::invalid_argument("Nonfinite priority");
        }
        if (fixed &&
            (tasks[i].assignment < 0 ||
             static_cast<std::size_t>(tasks[i].assignment) > graph.k)) {
            throw std::invalid_argument("Invalid fixed assignment");
        }
        reset_timing(tasks[i]);
        remaining[i] = graph.pred[i].size();
        if (remaining[i] == 0) {
            ready.push(i);
        }
    }

    while (!ready.empty()) {
        const std::size_t i = ready.top();
        ready.pop();
        auto& task = tasks[i];
        dependency_times(task, tasks, graph.pred[i]);

        const int fixed_assignment = task.assignment;
        const bool cloud_only = !task.is_core_task;
        Time best_finish = std::numeric_limits<Time>::infinity();
        Time best_start = 0;
        std::size_t best_unit = graph.k;

        if (!cloud_only || !cloud_enabled) {
            for (std::size_t core = 0; core < graph.k; ++core) {
                if (fixed && fixed_assignment != static_cast<int>(core)) {
                    continue;
                }
                const Time start = earliest_slot(
                    calendars[core], task.RT_l, task.core_execution_times[core]);
                const Time finish = add(start, task.core_execution_times[core]);
                // Equal finish times prefer the lower local core index.
                if (finish < best_finish) {
                    best_finish = finish;
                    best_start = start;
                    best_unit = core;
                }
            }
        }

        if (cloud_enabled &&
            (!fixed || fixed_assignment == static_cast<int>(graph.k))) {
            const Time start = earliest_slot(calendars[graph.k], task.RT_ws,
                                             task.cloud_execution_times[0]);
            const CloudTiming timing =
                calculate_cloud_timing(task, start, tasks, graph.pred[i]);
            // Equal local/cloud finish times prefer local execution.
            if (timing.return_finish < best_finish) {
                best_finish = timing.return_finish;
                best_start = start;
                best_unit = graph.k;
            }
        }

        if (!std::isfinite(best_finish)) {
            throw std::invalid_argument("No permitted execution location");
        }
        if (best_unit == graph.k) {
            place_cloud(task, graph.k, best_start, tasks, graph.pred[i]);
        } else {
            place_local(task, best_unit, best_start);
        }
        task.is_scheduled = SchedulingState::SCHEDULED;

        const Time resource_finish = best_unit == graph.k ? task.FT_ws : task.FT_l;
        insert_interval(calendars[best_unit],
                        {best_start, resource_finish, task.id});
        for (const std::size_t child : graph.succ[i]) {
            if (--remaining[child] == 0) {
                ready.push(child);
            }
        }
    }

    Sequences result(graph.k + 1);
    for (std::size_t resource = 0; resource <= graph.k; ++resource) {
        result[resource].reserve(calendars[resource].size());
        for (const auto& interval : calendars[resource]) {
            result[resource].push_back(interval.id);
        }
    }
    return result;
}

Sequences initial_schedule_impl(std::vector<Task>& tasks, const Graph& graph,
                                bool cloud_enabled) {
    primary_assignment_impl(tasks, graph, cloud_enabled);
    task_prioritizing_impl(tasks, graph);
    return select_units(tasks, graph, cloud_enabled, false);
}

void check_sequences(const std::vector<Task>& tasks, const Graph& graph,
                     const Sequences& sequences) {
    if (sequences.size() != graph.k + 1) {
        throw InvalidSequence("Expected K local sequences plus one upload sequence");
    }

    std::vector<std::uint8_t> seen(tasks.size(), 0);
    std::size_t seen_count = 0;
    for (std::size_t resource = 0; resource < sequences.size(); ++resource) {
        for (const int id : sequences[resource]) {
            const auto found = graph.index.find(id);
            if (found == graph.index.end() || seen[found->second] != 0) {
                throw InvalidSequence("Unknown or repeated task in sequences");
            }
            seen[found->second] = 1;
            ++seen_count;
            if (tasks[found->second].assignment != static_cast<int>(resource)) {
                throw InvalidSequence("Sequence/assignment mismatch");
            }
        }
    }
    if (seen_count != tasks.size()) {
        throw InvalidSequence("Task missing from sequences");
    }
}

void check_chronological_schedule(const std::vector<Task>& tasks, const Graph& graph,
                                  const Sequences& sequences) {
    for (std::size_t resource = 0; resource < sequences.size(); ++resource) {
        Time previous_finish = 0;
        for (const int id : sequences[resource]) {
            const auto& task = tasks[graph.index.at(id)];
            const Time finish = resource == graph.k ? task.FT_ws : task.FT_l;
            if (task.is_scheduled == SchedulingState::UNSCHEDULED ||
                !std::isfinite(task.execution_start_time) || !std::isfinite(finish) ||
                task.execution_start_time < previous_finish ||
                finish < task.execution_start_time) {
                throw InvalidSequence(
                    "Migration requires a complete chronological old schedule");
            }
            previous_finish = finish;
        }
    }
}

Sequences construct_sequence_impl(std::vector<Task>& tasks, std::size_t task_index,
                                  std::size_t destination, Sequences sequences,
                                  const Graph& graph) {
    auto& task = tasks[task_index];
    if (static_cast<std::size_t>(task.assignment) == destination) {
        return sequences;
    }

    Time ready = 0;
    for (const std::size_t predecessor : graph.pred[task_index]) {
        const Time transferred_finish =
            destination == graph.k ? tasks[predecessor].FT_ws : tasks[predecessor].FT_wr;
        ready = std::max(ready, std::max(tasks[predecessor].FT_l, transferred_finish));
    }

    auto& old_sequence = sequences[static_cast<std::size_t>(task.assignment)];
    const auto old_position =
        std::find(old_sequence.begin(), old_sequence.end(), task.id);
    if (old_position == old_sequence.end()) {
        throw InvalidSequence("Migrated task is missing from its resource sequence");
    }
    old_sequence.erase(old_position);

    auto& target_sequence = sequences[destination];
    const auto target_position = std::lower_bound(
        target_sequence.begin(), target_sequence.end(), ready,
        [&](int id, Time time) {
            const auto& other = tasks[graph.index.at(id)];
            if (other.is_scheduled == SchedulingState::UNSCHEDULED) {
                throw InvalidSequence("Missing old schedule timings");
            }
            return other.execution_start_time < time;
        });
    target_sequence.insert(target_position, task.id);

    task.assignment = static_cast<int>(destination);
    task.is_core_task = destination != graph.k;
    return sequences;
}

struct KernelWorkspace {
    std::vector<std::size_t> remaining;
    std::vector<std::size_t> next;
    std::vector<std::uint8_t> state;
    std::vector<std::size_t> stack;
    std::vector<Time> available;

    void reset(std::size_t task_count, std::size_t resource_count) {
        remaining.resize(task_count);
        next.assign(task_count, task_count);
        state.assign(task_count, 0);
        stack.clear();
        if (stack.capacity() < task_count) {
            stack.reserve(task_count);
        }
        available.assign(resource_count, 0);
    }
};

std::vector<Task>& kernel_algorithm_impl(std::vector<Task>& tasks,
                                         const Sequences& sequences,
                                         const Graph& graph,
                                         KernelStatistics* statistics,
                                         bool validate_sequences,
                                         KernelWorkspace* reusable_workspace = nullptr) {
    if (validate_sequences) {
        check_sequences(tasks, graph, sequences);
    }

    constexpr std::uint8_t sequence_ready_flag = 1U << 0U;
    constexpr std::uint8_t enqueued_flag = 1U << 1U;

    KernelStatistics stats;
    const std::size_t n = tasks.size();
    KernelWorkspace local_workspace;
    KernelWorkspace& workspace =
        reusable_workspace == nullptr ? local_workspace : *reusable_workspace;
    workspace.reset(n, graph.k + 1);

    for (std::size_t i = 0; i < n; ++i) {
        reset_timing(tasks[i]);
        workspace.remaining[i] = graph.pred[i].size();
    }
    for (const auto& sequence : sequences) {
        if (sequence.empty()) {
            continue;
        }
        workspace.state[graph.index.at(sequence.front())] |= sequence_ready_flag;
        for (std::size_t position = 1; position < sequence.size(); ++position) {
            workspace.next[graph.index.at(sequence[position - 1])] =
                graph.index.at(sequence[position]);
        }
    }

    const auto enqueue = [&](std::size_t index) {
        if (workspace.remaining[index] == 0 &&
            (workspace.state[index] & sequence_ready_flag) != 0 &&
            (workspace.state[index] & enqueued_flag) == 0) {
            workspace.stack.push_back(index);
            workspace.state[index] |= enqueued_flag;
        }
    };
    for (std::size_t i = 0; i < n; ++i) {
        enqueue(i);
    }

    while (!workspace.stack.empty()) {
        const std::size_t i = workspace.stack.back();
        workspace.stack.pop_back();
        auto& task = tasks[i];
        dependency_times(task, tasks, graph.pred[i]);

        const std::size_t resource = static_cast<std::size_t>(task.assignment);
        if (resource == graph.k) {
            place_cloud(task, graph.k,
                        std::max(workspace.available[resource], task.RT_ws), tasks,
                        graph.pred[i]);
            workspace.available[resource] = task.FT_ws;
        } else {
            place_local(task, resource,
                        std::max(workspace.available[resource], task.RT_l));
            workspace.available[resource] = task.FT_l;
        }
        task.is_scheduled = SchedulingState::KERNEL_SCHEDULED;
        ++stats.scheduled_tasks;

        for (const std::size_t child : graph.succ[i]) {
            --workspace.remaining[child];
            ++stats.dag_edge_updates;
            enqueue(child);
        }
        if (workspace.next[i] != n) {
            workspace.state[workspace.next[i]] |= sequence_ready_flag;
            ++stats.sequence_edge_updates;
            enqueue(workspace.next[i]);
        }
    }

    if (statistics != nullptr) {
        *statistics = stats;
    }
    if (stats.scheduled_tasks != n) {
        throw InvalidSequence("Resource sequences conflict with DAG precedence");
    }
    return tasks;
}

std::vector<TaskSchedule> capture_schedule_state(const std::vector<Task>& tasks) {
    std::vector<TaskSchedule> snapshot;
    snapshot.reserve(tasks.size());
    for (const auto& task : tasks) {
        snapshot.push_back(task);
    }
    return snapshot;
}

class ScopedScheduleRestore {
public:
    ScopedScheduleRestore(std::vector<Task>& tasks,
                          const std::vector<TaskSchedule>& snapshot)
        : tasks_(tasks), snapshot_(snapshot) {}

    ScopedScheduleRestore(const ScopedScheduleRestore&) = delete;
    ScopedScheduleRestore& operator=(const ScopedScheduleRestore&) = delete;

    ~ScopedScheduleRestore() {
        for (std::size_t i = 0; i < tasks_.size(); ++i) {
            static_cast<TaskSchedule&>(tasks_[i]) = snapshot_[i];
        }
    }

private:
    std::vector<Task>& tasks_;
    const std::vector<TaskSchedule>& snapshot_;
};
}  // namespace

// Task graph and model utilities

Task::Task(int task_id, std::vector<Time> local_times,
           std::array<Time, 3> remote_times)
    : id(task_id),
      core_execution_times(std::move(local_times)),
      cloud_execution_times(remote_times) {}

void validate_task_graph(const std::vector<Task>& tasks) {
    (void)Graph(tasks);
}

std::size_t core_count(const std::vector<Task>& tasks) {
    return Graph(tasks).k;
}

std::vector<Task> create_task_graph(const std::vector<int>& ids,
                                    const std::map<int, std::vector<Time>>& local_times,
                                    const std::array<Time, 3>& cloud_times,
                                    const std::vector<std::pair<int, int>>& edges) {
    std::vector<Task> tasks;
    tasks.reserve(ids.size());

    std::unordered_map<int, std::size_t> index;
    index.max_load_factor(0.7F);
    index.reserve(ids.size());
    for (const int id : ids) {
        const auto durations = local_times.find(id);
        if (durations == local_times.end()) {
            throw std::invalid_argument("Missing task durations");
        }
        if (!index.emplace(id, tasks.size()).second) {
            throw std::invalid_argument("Duplicate task ID");
        }
        tasks.emplace_back(id, durations->second, cloud_times);
    }

    for (const auto& edge : edges) {
        if (index.count(edge.first) == 0 || index.count(edge.second) == 0) {
            throw std::invalid_argument("Unknown edge endpoint");
        }
        tasks[index.at(edge.first)].succ_tasks.push_back(edge.second);
        tasks[index.at(edge.second)].pred_tasks.push_back(edge.first);
    }

    validate_task_graph(tasks);
    return tasks;
}

Time total_time(const std::vector<Task>& tasks) {
    Time result = 0;
    for (const auto& task : tasks) {
        if (task.succ_tasks.empty()) {
            result = std::max(result, std::max(task.FT_l, task.FT_wr));
        }
    }
    return result;
}

Time transmission_time(double data_amount, double data_rate) {
    if (!std::isfinite(data_amount) || data_amount < 0 ||
        !std::isfinite(data_rate) || data_rate <= 0) {
        throw std::invalid_argument("Data must be finite/nonnegative and rate finite/positive");
    }
    const Time result = data_amount / data_rate;
    if (!std::isfinite(result) || (data_amount > 0 && result == 0)) {
        throw std::overflow_error("Communication duration is not representable");
    }
    return result;
}

// Energy model

std::vector<double> default_core_powers(std::size_t count) {
    std::vector<double> result(count);
    double power = 1;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(power)) {
            throw std::invalid_argument(
                "Default core powers overflow; provide explicit powers");
        }
        result[i] = power;
        power *= 2;
    }
    return result;
}

double calculate_energy_consumption(const Task& task,
                                    const std::vector<double>& powers,
                                    double sending_power) {
    check_energy(task.core_execution_times.size(), powers, sending_power);
    return energy_value(task, powers, sending_power);
}

double total_energy(const std::vector<Task>& tasks, const std::vector<double>& powers,
                    double sending_power) {
    const std::size_t cores =
        tasks.empty() ? powers.size() : tasks.front().core_execution_times.size();
    check_energy(cores, powers, sending_power);

    double result = 0;
    for (const auto& task : tasks) {
        result += energy_value(task, powers, sending_power);
    }
    if (!std::isfinite(result)) {
        throw std::overflow_error("Total energy overflow");
    }
    return result;
}

// Initial scheduling public API

void primary_assignment(std::vector<Task>& tasks, std::size_t expected_cores,
                        bool cloud_enabled) {
    const Graph graph(tasks);
    if (expected_cores != 0 && expected_cores != graph.k) {
        throw std::invalid_argument("Requested core count differs from task data");
    }
    primary_assignment_impl(tasks, graph, cloud_enabled);
}

void task_prioritizing(std::vector<Task>& tasks) {
    const Graph graph(tasks);
    task_prioritizing_impl(tasks, graph);
}

Sequences execution_unit_selection(std::vector<Task>& tasks, bool cloud_enabled) {
    const Graph graph(tasks);
    return select_units(tasks, graph, cloud_enabled, false);
}

Sequences initial_schedule(std::vector<Task>& tasks, bool cloud_enabled) {
    const Graph graph(tasks);
    return initial_schedule_impl(tasks, graph, cloud_enabled);
}

Sequences fixed_assignment_schedule(std::vector<Task>& tasks,
                                    const std::vector<int>& assignments) {
    const Graph graph(tasks);
    if (assignments.size() != tasks.size()) {
        throw std::invalid_argument("Expected one fixed assignment per task");
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (assignments[i] < 0 ||
            static_cast<std::size_t>(assignments[i]) > graph.k) {
            throw std::invalid_argument("Invalid fixed assignment");
        }
        tasks[i].assignment = assignments[i];
        tasks[i].is_core_task =
            static_cast<std::size_t>(assignments[i]) != graph.k;
    }
    task_prioritizing_impl(tasks, graph);
    return select_units(tasks, graph, true, true);
}

// Migration and incremental rescheduling

Sequences construct_sequence(std::vector<Task>& tasks, int task_id, int destination,
                             Sequences sequences) {
    Graph graph(tasks, false);
    check_sequences(tasks, graph, sequences);
    if (destination < 0 || static_cast<std::size_t>(destination) > graph.k ||
        !graph.index.count(task_id)) {
        throw std::invalid_argument("Invalid migration target");
    }
    check_chronological_schedule(tasks, graph, sequences);
    return construct_sequence_impl(tasks, graph.index.at(task_id),
                                   static_cast<std::size_t>(destination),
                                   std::move(sequences), graph);
}

std::vector<Task>& kernel_algorithm(std::vector<Task>& tasks,
                                    const Sequences& sequences,
                                    KernelStatistics* statistics) {
    Graph graph(tasks, false);
    return kernel_algorithm_impl(tasks, sequences, graph, statistics, true);
}

std::pair<std::vector<Task>, Sequences> optimize_task_scheduling(
    std::vector<Task> tasks, Sequences sequences, Time deadline,
    std::vector<double> powers, double sending_power, bool cloud_enabled,
    MigrationStatistics* statistics) {
    check_deadline(deadline);
    const Graph graph(tasks);
    check_sequences(tasks, graph, sequences);
    if (powers.empty()) {
        powers = default_core_powers(graph.k);
    }
    check_energy(graph.k, powers, sending_power);

    const auto checked = validate_schedule_constraints(
        tasks, std::numeric_limits<Time>::infinity(), &sequences);
    if (!std::get<0>(checked)) {
        throw std::invalid_argument(
            "Optimization requires a valid complete initial schedule: " +
            std::get<1>(checked).front());
    }
    if (!cloud_enabled && !sequences.back().empty()) {
        throw std::invalid_argument("Local-only scheduling cannot contain cloud tasks");
    }
    if (exceeds(total_time(tasks), deadline)) {
        throw DeadlineNotMet(
            "Initial heuristic exceeds deadline; this is not a proof of infeasibility");
    }

    MigrationStatistics stats;
    std::vector<std::size_t> order(tasks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return tasks[left].id < tasks[right].id;
    });
    KernelWorkspace kernel_workspace;

    for (;;) {
        const Time current_time = total_time(tasks);
        const double current_energy = total_energy(tasks, powers, sending_power);
        check_chronological_schedule(tasks, graph, sequences);

        bool found = false;
        bool best_no_increase = false;
        double best_saving = 0;
        long double best_ratio = 0;
        std::size_t best_task = 0;
        std::size_t best_destination = 0;
        const auto original = capture_schedule_state(tasks);

        for (const std::size_t task_index : order) {
            // The paper explicitly excludes cloud-to-local migration.
            if (static_cast<std::size_t>(tasks[task_index].assignment) == graph.k) {
                continue;
            }
            const std::size_t destination_count = graph.k + (cloud_enabled ? 1 : 0);
            for (std::size_t destination = 0; destination < destination_count;
                 ++destination) {
                if (destination ==
                    static_cast<std::size_t>(tasks[task_index].assignment)) {
                    continue;
                }

                ++stats.candidates;
                ScopedScheduleRestore restore(tasks, original);
                auto trial_sequences = construct_sequence_impl(
                    tasks, task_index, destination, sequences, graph);
                try {
                    kernel_algorithm_impl(tasks, trial_sequences, graph, nullptr, false,
                                          &kernel_workspace);
                } catch (const InvalidSequence&) {
                    ++stats.rejected_sequences;
                    continue;
                }

                const Time trial_time = total_time(tasks);
                const double saving =
                    current_energy - total_energy(tasks, powers, sending_power);
                if (exceeds(trial_time, deadline) || saving <= 0) {
                    continue;
                }

                const bool no_increase = trial_time <= current_time;
                const long double ratio =
                    no_increase
                        ? 0
                        : static_cast<long double>(saving) /
                              static_cast<long double>(trial_time - current_time);
                const bool better =
                    !found || (no_increase && !best_no_increase) ||
                    (no_increase && best_no_increase && saving > best_saving) ||
                    (!no_increase && !best_no_increase && ratio > best_ratio);
                if (better) {
                    found = true;
                    best_no_increase = no_increase;
                    best_saving = saving;
                    best_ratio = ratio;
                    best_task = task_index;
                    best_destination = destination;
                }
            }
        }

        if (!found) {
            break;
        }

        sequences = construct_sequence_impl(tasks, best_task, best_destination,
                                            std::move(sequences), graph);
        kernel_algorithm_impl(tasks, sequences, graph, nullptr, false,
                              &kernel_workspace);
        if (!(total_energy(tasks, powers, sending_power) < current_energy) ||
            exceeds(total_time(tasks), deadline)) {
            throw std::logic_error("Accepted migration failed energy/deadline invariant");
        }
        ++stats.accepted;
    }

    if (statistics != nullptr) {
        *statistics = stats;
    }
    return {std::move(tasks), std::move(sequences)};
}
ScheduleResult schedule_application(std::vector<Task> tasks, Time deadline,
                                    EnergyModel energy, bool cloud_enabled) {
    check_deadline(deadline);
    const Graph graph(tasks);
    const std::size_t k = graph.k;
    if (energy.core_powers.empty()) {
        energy.core_powers = default_core_powers(k);
    }
    check_energy(k, energy.core_powers, energy.sending_power);

    ScheduleResult result;
    result.deadline = deadline;
    result.initial_sequences = initial_schedule_impl(tasks, graph, cloud_enabled);
    result.initial_tasks = tasks;
    result.tasks = tasks;
    result.sequences = result.initial_sequences;
    if (exceeds(total_time(tasks), deadline)) {
        return result;
    }

    auto final = optimize_task_scheduling(
        std::move(tasks), result.sequences, deadline, energy.core_powers,
        energy.sending_power, cloud_enabled, &result.migrations);
    result.tasks = std::move(final.first);
    result.sequences = std::move(final.second);
    result.feasible = true;
    return result;
}

// Independent schedule validation

std::tuple<bool, std::vector<std::string>> validate_schedule_constraints(
    const std::vector<Task>& tasks, Time deadline, const Sequences* sequences) {
    std::vector<std::string> errors;
    try {
        const Graph graph(tasks);
        if (std::isnan(deadline) || deadline < 0) {
            errors.push_back("Invalid deadline");
        }

        std::vector<std::vector<Interval>> intervals(graph.k + 1);
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            const auto& task = tasks[i];
            const auto add_task_error = [&](const std::string& message) {
                errors.push_back("Task " + std::to_string(task.id) + ": " + message);
            };

            if (task.assignment < 0 ||
                static_cast<std::size_t>(task.assignment) > graph.k) {
                add_task_error("missing/invalid assignment");
                continue;
            }
            const std::size_t resource = static_cast<std::size_t>(task.assignment);
            const bool local = resource < graph.k;
            if (task.is_core_task != local) {
                add_task_error("assignment/type mismatch");
            }
            if (task.is_scheduled == SchedulingState::UNSCHEDULED) {
                add_task_error("task was not scheduled");
            }

            for (const Time value : {task.RT_l, task.RT_ws, task.RT_c, task.RT_wr,
                                     task.FT_l, task.FT_ws, task.FT_c, task.FT_wr,
                                     task.execution_finish_time}) {
                if (!std::isfinite(value) || value < 0) {
                    add_task_error("nonfinite or negative timing");
                }
            }
            const Time start = task.execution_start_time;
            if (!std::isfinite(start) || start < 0) {
                add_task_error("invalid start time");
            }

            Time local_ready = 0;
            Time upload_ready = 0;
            Time cloud_parent_finish = 0;
            for (const std::size_t predecessor : graph.pred[i]) {
                local_ready = std::max(
                    {local_ready, tasks[predecessor].FT_l, tasks[predecessor].FT_wr});
                upload_ready = std::max(
                    {upload_ready, tasks[predecessor].FT_l, tasks[predecessor].FT_ws});
                cloud_parent_finish =
                    std::max(cloud_parent_finish, tasks[predecessor].FT_c);
            }
            if (!close(task.RT_l, local_ready)) {
                add_task_error("Equation 3 ready time mismatch");
            }
            if (!close(task.RT_ws, upload_ready)) {
                add_task_error("Equation 4 ready time mismatch");
            }

            if (local) {
                if (exceeds(local_ready, start)) {
                    add_task_error("local predecessor not complete");
                }
                if (!close(task.FT_l,
                           start + task.core_execution_times[resource])) {
                    add_task_error("local duration mismatch");
                }
                if (task.FT_ws != 0 || task.FT_c != 0 || task.FT_wr != 0 ||
                    task.RT_c != 0 || task.RT_wr != 0) {
                    add_task_error("inactive cloud timings must be zero");
                }
                if (!close(task.execution_finish_time, task.FT_l)) {
                    add_task_error("overall finish mismatch");
                }
                if (std::isfinite(start) && std::isfinite(task.FT_l) && start >= 0 &&
                    task.FT_l >= 0) {
                    intervals[resource].push_back({start, task.FT_l, task.id});
                }
            } else {
                if (exceeds(upload_ready, start)) {
                    add_task_error("upload predecessor not ready");
                }
                if (!close(task.FT_ws, start + task.cloud_execution_times[0])) {
                    add_task_error("upload duration mismatch");
                }
                if (!close(task.RT_c, std::max(task.FT_ws, cloud_parent_finish))) {
                    add_task_error("Equation 5 cloud ready time mismatch");
                }
                if (!close(task.FT_c,
                           task.RT_c + task.cloud_execution_times[1])) {
                    add_task_error("cloud duration/parallel start mismatch");
                }
                if (!close(task.RT_wr, task.FT_c) ||
                    !close(task.FT_wr,
                           task.FT_c + task.cloud_execution_times[2])) {
                    add_task_error("Equation 6 immediate return mismatch");
                }
                if (task.FT_l != 0) {
                    add_task_error("inactive local finish must be zero");
                }
                if (!close(task.execution_finish_time, task.FT_wr)) {
                    add_task_error("overall finish mismatch");
                }
                if (std::isfinite(start) && std::isfinite(task.FT_ws) && start >= 0 &&
                    task.FT_ws >= 0) {
                    intervals[resource].push_back({start, task.FT_ws, task.id});
                }
            }
        }

        for (std::size_t resource = 0; resource < intervals.size(); ++resource) {
            auto& list = intervals[resource];
            std::sort(list.begin(), list.end(),
                      [](const Interval& left, const Interval& right) {
                          return left.start < right.start;
                      });
            for (std::size_t i = 1; i < list.size(); ++i) {
                if (exceeds(list[i - 1].finish, list[i].start)) {
                    errors.push_back(
                        (resource == graph.k ? "Upload conflict: " : "Core conflict: ") +
                        std::to_string(list[i - 1].id) + " / " +
                        std::to_string(list[i].id));
                }
            }
        }

        if (sequences != nullptr) {
            check_sequences(tasks, graph, *sequences);
            for (std::size_t resource = 0; resource < sequences->size(); ++resource) {
                Time finish = 0;
                for (const int id : (*sequences)[resource]) {
                    const auto& task = tasks[graph.index.at(id)];
                    if (exceeds(finish, task.execution_start_time)) {
                        errors.push_back("Timing disagrees with resource sequence");
                    }
                    finish = resource == graph.k ? task.FT_ws : task.FT_l;
                }
            }
        }

        if (exceeds(total_time(tasks), deadline)) {
            errors.push_back("Deadline exceeded");
        }
    } catch (const std::exception& error) {
        errors.push_back(error.what());
    }
    return {errors.empty(), errors};
}

// Reporting

void print_schedule_tasks(const std::vector<Task>& tasks) {
    std::cout << "Task  Resource  Start  Local finish  Upload finish  "
                 "Cloud start/finish  Return start/finish\n";
    for (const auto& task : tasks) {
        const std::size_t k = task.core_execution_times.size();
        const std::string resource =
            static_cast<std::size_t>(task.assignment) == k
                ? "Cloud"
                : "Core " + std::to_string(task.assignment + 1);
        std::cout << task.id << "  " << resource << "  " << task.execution_start_time
                  << "  " << task.FT_l << "  " << task.FT_ws << "  " << task.RT_c
                  << "/" << task.FT_c << "  " << task.RT_wr << "/" << task.FT_wr
                  << '\n';
    }
}

void print_schedule_sequences(const Sequences& sequences) {
    for (std::size_t resource = 0; resource < sequences.size(); ++resource) {
        const std::string name = resource + 1 == sequences.size()
                                     ? "Upload"
                                     : "Core " + std::to_string(resource + 1);
        std::cout << name << ':';
        for (const int id : sequences[resource]) {
            std::cout << ' ' << id;
        }
        std::cout << '\n';
    }
}

void print_schedule_validation_report(const std::vector<Task>& tasks, Time deadline) {
    const auto [valid, errors] = validate_schedule_constraints(tasks, deadline);
    if (valid) {
        std::cout << "PASS: paper-model schedule constraints\n";
        return;
    }
    for (const auto& error : errors) {
        std::cout << "FAIL: " << error << '\n';
    }
}
