#include "mcc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Lightweight test harness

std::size_t checks = 0;

void require(bool condition, std::string_view message) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void near(Time actual, Time expected, std::string_view message) {
    ++checks;
    const Time tolerance = 1e-10 * std::max(1.0, std::abs(expected));
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(std::string(message) + ": got " +
                                 std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

template <class Exception = std::invalid_argument, class Function>
void throws(Function&& function, std::string_view message) {
    bool caught = false;
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, message);
}

void valid(const std::vector<Task>& tasks, const Sequences& sequences,
           Time deadline = std::numeric_limits<Time>::infinity()) {
    const auto [ok, errors] =
        validate_schedule_constraints(tasks, deadline, &sequences);
    ++checks;
    if (ok) {
        return;
    }

    std::string details = "Paper-model validation failed";
    for (const auto& error : errors) {
        details += '\n';
        details += error;
    }
    throw std::runtime_error(details);
}

// Independent expanded-event timing oracle ---------------------------------

struct Event {
    Time duration = 0;
    Time start = 0;
    std::size_t indegree = 0;
};

struct EventEdge {
    std::size_t source;
    std::size_t destination;
};

std::size_t estimate_event_edges(const std::vector<Task>& tasks,
                                 const Sequences& sequences,
                                 std::size_t cloud_resource) {
    std::size_t count = 0;
    for (const auto& task : tasks) {
        if (task.assignment == static_cast<int>(cloud_resource)) {
            count += 2;
        }
        count += 2 * task.pred_tasks.size();
    }
    for (const auto& sequence : sequences) {
        if (!sequence.empty()) {
            count += sequence.size() - 1;
        }
    }
    return count;
}

std::vector<Event> event_oracle(const std::vector<Task>& tasks,
                                const Sequences& sequences) {
    const std::size_t cores = tasks.front().core_execution_times.size();

    std::unordered_map<int, std::size_t> index;
    index.reserve(tasks.size());
    index.max_load_factor(0.7F);
    for (std::size_t position = 0; position < tasks.size(); ++position) {
        index.emplace(tasks[position].id, position);
    }

    std::vector<Event> events(tasks.size() * 3);
    std::vector<EventEdge> edges;
    edges.reserve(estimate_event_edges(tasks, sequences, cores));
    const auto add_edge = [&](std::size_t source, std::size_t destination) {
        edges.push_back({source, destination});
        ++events[destination].indegree;
    };

    for (std::size_t position = 0; position < tasks.size(); ++position) {
        const Task& task = tasks[position];
        const bool cloud = task.assignment == static_cast<int>(cores);
        const std::size_t base = 3 * position;

        if (cloud) {
            for (std::size_t phase = 0; phase < 3; ++phase) {
                events[base + phase].duration =
                    task.cloud_execution_times[phase];
            }
            add_edge(base, base + 1);
            add_edge(base + 1, base + 2);
        } else {
            events[base].duration =
                task.core_execution_times.at(
                    static_cast<std::size_t>(task.assignment));
        }

        for (int predecessor_id : task.pred_tasks) {
            const std::size_t predecessor = index.at(predecessor_id);
            const bool predecessor_cloud =
                tasks[predecessor].assignment == static_cast<int>(cores);
            const std::size_t predecessor_base = 3 * predecessor;

            if (!cloud) {
                add_edge(predecessor_base + (predecessor_cloud ? 2 : 0), base);
            } else {
                add_edge(predecessor_base, base);
                if (predecessor_cloud) {
                    add_edge(predecessor_base + 1, base + 1);
                }
            }
        }
    }

    for (const auto& sequence : sequences) {
        for (std::size_t position = 1; position < sequence.size(); ++position) {
            add_edge(3 * index.at(sequence[position - 1]),
                     3 * index.at(sequence[position]));
        }
    }

    // Compact sparse row storage keeps the oracle independent while avoiding
    // one dynamically allocated successor vector per event.
    std::vector<std::size_t> offsets(events.size() + 1, 0);
    for (const EventEdge& edge : edges) {
        ++offsets[edge.source + 1];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());

    std::vector<std::size_t> successors(edges.size());
    std::vector<std::size_t> cursor = offsets;
    for (const EventEdge& edge : edges) {
        successors[cursor[edge.source]++] = edge.destination;
    }

    std::vector<std::size_t> ready;
    ready.reserve(events.size());
    for (std::size_t event = 0; event < events.size(); ++event) {
        if (events[event].indegree == 0) {
            ready.push_back(event);
        }
    }

    for (std::size_t position = 0; position < ready.size(); ++position) {
        const std::size_t event = ready[position];
        const Time finish = events[event].start + events[event].duration;
        for (std::size_t edge = offsets[event]; edge < offsets[event + 1];
             ++edge) {
            Event& child = events[successors[edge]];
            child.start = std::max(child.start, finish);
            if (--child.indegree == 0) {
                ready.push_back(successors[edge]);
            }
        }
    }

    if (ready.size() != events.size()) {
        throw InvalidSequence("Independent event DAG contains a cycle");
    }
    return events;
}

void matches_oracle(const std::vector<Task>& tasks,
                    const Sequences& sequences) {
    const std::vector<Event> events = event_oracle(tasks, sequences);
    const std::size_t cores = tasks.front().core_execution_times.size();

    for (std::size_t position = 0; position < tasks.size(); ++position) {
        const Task& task = tasks[position];
        const std::size_t base = 3 * position;
        near(task.execution_start_time, events[base].start,
             "Oracle resource start");

        if (task.assignment == static_cast<int>(cores)) {
            near(task.FT_ws, events[base].start + events[base].duration,
                 "Oracle upload finish");
            near(task.RT_c, events[base + 1].start, "Oracle cloud start");
            near(task.FT_c,
                 events[base + 1].start + events[base + 1].duration,
                 "Oracle cloud finish");
            near(task.RT_wr, events[base + 2].start,
                 "Oracle receive start");
            near(task.FT_wr,
                 events[base + 2].start + events[base + 2].duration,
                 "Oracle receive finish");
        } else {
            near(task.FT_l, events[base].start + events[base].duration,
                 "Oracle local finish");
        }
    }
}

// Paper equations and figures

void paper_examples() {
    near(transmission_time(6, 2), 3, "Equation 1 upload from data/rate");
    near(transmission_time(2, 2), 1, "Equation 2 receive from data/rate");
    near(transmission_time(0, 2), 0, "Empty result transfer");
    throws([&] { transmission_time(1, 0); }, "Zero transfer rate rejected");
    throws([&] { transmission_time(-1, 2); }, "Negative data rejected");

    auto tasks = example_graphs().front();
    primary_assignment(tasks);
    for (const auto& task : tasks) {
        require(task.is_core_task,
                "Figure 1 primary assignment uses strict inequality");
    }

    task_prioritizing(tasks);
    near(tasks[0].priority_score, 65.0 / 3, "Equation 15 root priority");
    near(tasks[9].priority_score, 13.0 / 3, "Equation 16 exit priority");

    Sequences sequences = execution_unit_selection(tasks);
    const Sequences expected = {{4}, {6, 8}, {1, 3, 5, 7, 9, 10}, {2}};
    require(sequences == expected, "Figure 3 execution sequences");

    const std::array<Time, 10> starts = {0, 5, 5, 5, 9, 5, 11, 12, 14, 16};
    const std::array<Time, 10> finishes = {5, 10, 9, 12, 11,
                                           11, 14, 16, 16, 18};
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        near(tasks[index].execution_start_time, starts[index],
             "Figure 3 start");
        near(tasks[index].execution_finish_time, finishes[index],
             "Figure 3 finish");
    }
    near(total_time(tasks), 18, "Figure 3 completion");
    near(total_energy(tasks, {1, 2, 4}, 0.5), 100.5, "Figure 3 energy");
    valid(tasks, sequences);
    matches_oracle(tasks, sequences);

    // Independently supplied Figure 4 assignments/order, not optimizer output.
    auto figure4 = example_graphs().front();
    for (auto& task : figure4) {
        task.assignment = task.id == 4 ? 0 : (task.id == 9 ? 2 : 3);
    }
    const Sequences figure4_sequences = {
        {4}, {}, {9}, {1, 2, 5, 6, 3, 8, 7, 10}};
    kernel_algorithm(figure4, figure4_sequences);
    near(total_time(figure4), 26, "Figure 4 completion");
    near(total_energy(figure4, {1, 2, 4}, 0.5), 27,
         "Figure 4 energy");
    valid(figure4, figure4_sequences, 27);
    matches_oracle(figure4, figure4_sequences);

    auto further = optimize_task_scheduling(figure4, figure4_sequences, 27);
    require(total_energy(further.first, {1, 2, 4}, 0.5) <= 24,
            "Figure 4 admits an extra beneficial move");
    valid(further.first, further.second, 27);
    matches_oracle(further.first, further.second);

    auto result = schedule_application(example_graphs().front(), 27);
    require(result.feasible, "Figure 1 explicit deadline feasible");
    require(total_energy(result.tasks, {1, 2, 4}, 0.5) <= 27,
            "Example energy decreases");
    valid(result.tasks, result.sequences, 27);
    matches_oracle(result.tasks, result.sequences);
}

// Timing, dependency, and insertion behavior --------------------------------

void cloud_parallelism_and_dependencies() {
    std::vector<Task> independent = {
        Task(10, {100}, {1, 10, 4}), Task(20, {100}, {1, 10, 4})};
    Sequences sequence = initial_schedule(independent);

    Task a = independent[0];
    Task b = independent[1];
    if (a.execution_start_time > b.execution_start_time) {
        std::swap(a, b);
    }
    near(a.execution_start_time, 0, "First upload");
    near(b.execution_start_time, 1, "Serialized second upload");
    near(a.RT_c, 1, "First cloud starts at own upload finish");
    near(b.RT_c, 2, "Second cloud does not wait for first");
    require(b.RT_c < a.FT_c, "Independent cloud computations overlap");
    require(b.RT_wr < a.FT_wr, "Independent downloads overlap");
    near(total_time(independent), 16, "Parallel cloud makespan");
    valid(independent, sequence);
    matches_oracle(independent, sequence);

    kernel_algorithm(independent, sequence);
    valid(independent, sequence);
    matches_oracle(independent, sequence);

    auto chain = create_task_graph(
        {10, 20, 30}, {{10, {100}}, {20, {100}}, {30, {1}}}, {1, 10, 4},
        {{10, 20}, {20, 30}});
    chain[1].cloud_execution_times = {1, 1, 1};
    Sequences chain_sequences = initial_schedule(chain);
    near(chain[1].execution_start_time, 1,
         "Cloud child uploads while parent computes");
    near(chain[1].RT_c, 11,
         "Equation 5 waits for parent's long computation");
    near(chain[2].execution_start_time, 13,
         "Cloud-to-local waits for download");
    near(total_time(chain), 14, "Mixed dependency chain completion");
    valid(chain, chain_sequences);
    matches_oracle(chain, chain_sequences);

    kernel_algorithm(chain, chain_sequences);
    valid(chain, chain_sequences);
    matches_oracle(chain, chain_sequences);

    auto bad = chain;
    bad[1].RT_c = 2;
    bad[1].FT_c = 3;
    bad[1].RT_wr = 3;
    bad[1].FT_wr = 4;
    bad[1].execution_finish_time = 4;
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "Validator rejects missing cloud-parent dependency");

    bad = independent;
    bad[1].FT_wr += 10;
    bad[1].execution_finish_time += 10;
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "Validator rejects artificial receiver queue");
}

void initial_priority_and_gap_insertion() {
    // All tasks initially prefer local in isolation. Entry tasks must still be
    // compared with cloud when the core is occupied.
    std::vector<Task> entries = {
        Task(1, {5}, {3, 1, 1}), Task(2, {5}, {3, 1, 1})};
    Sequences sequence = initial_schedule(entries);
    require(sequence[0].size() == 1 && sequence[1].size() == 1,
            "Entry selection considers cloud contention tradeoff");
    near(total_time(entries), 5, "Two entries can finish together");
    valid(entries, sequence);
    matches_oracle(entries, sequence);

    // Root 1 -> child 2 ranks above independent entry 3. Child 2 reserves core 0
    // at [10,15]. Lower-ranked entry 3 then fills [0,2], not [15,17].
    auto gap = create_task_graph(
        {1, 2, 3},
        {{1, {100, 10}}, {2, {5, 100}}, {3, {2, 100}}},
        {100, 100, 100}, {{1, 2}});
    Sequences gap_sequences = fixed_assignment_schedule(gap, {1, 0, 0});
    near(gap[1].execution_start_time, 10,
         "High rank dependent scheduled before low-rank entry");
    near(gap[2].execution_start_time, 0, "Earliest local gap filled");
    require(gap_sequences[0] == std::vector<int>({3, 2}),
            "Sequence reflects chronological gap insertion");
    valid(gap, gap_sequences);
    matches_oracle(gap, gap_sequences);

    // Same situation on the serialized upload channel.
    auto upload = create_task_graph(
        {1, 2, 3}, {{1, {10}}, {2, {100}}, {3, {100}}}, {1, 1, 1},
        {{1, 2}});
    upload[1].cloud_execution_times = {2, 20, 1};
    Sequences uploads = fixed_assignment_schedule(upload, {0, 1, 1});
    near(upload[1].execution_start_time, 10, "Dependent upload ready time");
    near(upload[2].execution_start_time, 0, "Earlier upload gap filled");
    require(uploads[1] == std::vector<int>({3, 2}),
            "Chronological upload gap order");
    valid(upload, uploads);
    matches_oracle(upload, uploads);
}

// Migration and policy behavior

void migration_insertion_and_policy() {
    auto graph = create_task_graph(
        {10, 20, 30}, {{10, {100}}, {20, {5}}, {30, {100}}}, {1, 1, 1},
        {{10, 20}});
    graph[0].cloud_execution_times = {1, 10, 1};
    Sequences sequences = fixed_assignment_schedule(graph, {1, 0, 1});
    near(graph[1].RT_l, 12, "Old local ready");
    near(graph[1].RT_ws, 1, "Destination upload ready");

    // Prove reconstruction uses predecessor state, not stale ready fields.
    graph[1].RT_l = 900;
    graph[1].RT_ws = 999;
    Sequences moved = construct_sequence(graph, 20, 1, sequences);
    require(moved[1] == std::vector<int>({10, 20, 30}),
            "Equation 19 destination-specific insertion and equality tie");
    kernel_algorithm(graph, moved);
    valid(graph, moved);
    matches_oracle(graph, moved);

    auto task = create_task_graph({1}, {{1, {9, 7, 5}}}, {20, 1, 1}, {});
    Sequences initial = initial_schedule(task);
    MigrationStatistics stats;
    auto ratio = optimize_task_scheduling(task, initial, 9, {1, 2, 4}, 100,
                                          true, &stats);
    require(stats.accepted == 2,
            "Ratio selects core 2 before core 1 (3 versus 2.75 savings/time)");
    require(ratio.first[0].assignment == 0, "Final slower low-power core");
    near(total_energy(ratio.first, {1, 2, 4}, 100), 9,
         "Ratio policy final energy");

    auto prefer = create_task_graph(
        {1, 2}, {{1, {9, 7, 5}}, {2, {100, 200, 200}}}, {20, 1, 1}, {});
    prefer[1].cloud_execution_times = {1000, 1, 1};
    Sequences old = fixed_assignment_schedule(prefer, {2, 0});
    auto chosen = optimize_task_scheduling(prefer, old, 120);
    require(chosen.first[0].assignment == 3,
            "No-time-increase energy saving wins over larger saving with delay");
    near(total_time(chosen.first), 100, "Stage 1 preserves makespan");
    valid(chosen.first, chosen.second, 120);

    auto absorbing =
        create_task_graph({1}, {{1, {100}}}, {1, 1, 1}, {});
    Sequences cloud = initial_schedule(absorbing);
    auto keep = optimize_task_scheduling(absorbing, cloud, 100, {0.001}, 1000,
                                         true, &stats);
    require(keep.first[0].assignment == 1 && stats.candidates == 0,
            "Paper explicitly excludes cloud-to-local migration even if cheaper");
}

void configured_energy_and_deadlines() {
    auto task = create_task_graph({1}, {{1, {9, 7, 5}}}, {3, 1, 1}, {});
    Sequences sequences = initial_schedule(task);

    auto custom = optimize_task_scheduling(task, sequences, 7.5, {1, 2, 4}, 100);
    near(total_time(custom.first), 7, "Explicit fractional deadline");
    near(total_energy(custom.first, {1, 2, 4}, 100), 14,
         "Configured RF power applied in trials");

    auto core_powers =
        optimize_task_scheduling(task, sequences, 7.5, {1, 100, 1}, 100);
    require(core_powers.first[0].assignment == 2,
            "Configured core powers apply");

    auto tight = optimize_task_scheduling(task, sequences, 6.9,
                                          {1, 2, 4}, 100);
    near(total_time(tight.first), 5,
         "Actual deadline is not multiplied by 1.5");
    throws<DeadlineNotMet>(
        [&] { optimize_task_scheduling(task, sequences, 4); },
        "Impossible initial deadline reported");

    auto failed = schedule_application(task, 4);
    require(!failed.feasible && failed.migrations.accepted == 0,
            "No falsely feasible schedule on initial failure");

    auto large =
        create_task_graph({7}, {{7, {1e9 + 0.5}}}, {2e9, 1, 1}, {});
    auto result = schedule_application(large, 1e9);
    require(!result.feasible,
            "Hard deadline does not gain relative-tolerance slack");
    require(!std::get<0>(validate_schedule_constraints(result.tasks, 1e9)),
            "Validator enforces exact hard deadline");

    auto local = schedule_application(example_graphs().front(), 100, {}, false);
    require(local.feasible && local.sequences.back().empty(),
            "Baseline 2 never offloads");
    valid(local.tasks, local.sequences, 100);
    matches_oracle(local.tasks, local.sequences);
}

// Invalid models and schedules

void malformed_inputs_and_schedules() {
    throws([&] { create_task_graph({1, 1}, {{1, {1}}}, {1, 1, 1}, {}); },
           "Duplicate IDs");
    throws([&] { create_task_graph({1}, {{1, {1}}}, {1, 1, 1}, {{1, 2}}); },
           "Unknown edge");
    throws(
        [&] {
            create_task_graph({1, 2}, {{1, {1}}, {2, {1}}}, {1, 1, 1},
                              {{1, 2}, {2, 1}});
        },
        "DAG cycle");
    throws(
        [&] {
            create_task_graph({1, 2}, {{1, {1}}, {2, {1}}}, {1, 1, 1},
                              {{1, 2}, {1, 2}});
        },
        "Duplicate edges");
    throws([&] { create_task_graph({1}, {{1, {0}}}, {1, 1, 1}, {}); },
           "Nonpositive local duration");
    throws([&] { create_task_graph({1}, {{1, {1}}}, {1, -1, 1}, {}); },
           "Negative cloud duration");
    throws(
        [&] {
            create_task_graph({1, 2}, {{1, {1}}, {2, {1, 2}}}, {1, 1, 1},
                              {});
        },
        "Inconsistent K");
    throws([&] { create_task_graph({}, {}, {1, 1, 1}, {}); },
           "Empty application");

    auto graph = create_task_graph(
        {1, 2}, {{1, {1}}, {2, {1}}}, {10, 1, 1}, {{1, 2}});
    Sequences sequences = initial_schedule(graph);
    throws([&] { schedule_application(graph, 100, {{-1}, 0.5}); },
           "Negative power");
    throws(
        [&] {
            schedule_application(graph,
                                 std::numeric_limits<double>::quiet_NaN());
        },
        "NaN deadline");

    Sequences reversed = sequences;
    std::reverse(reversed[0].begin(), reversed[0].end());
    throws<InvalidSequence>([&] { kernel_algorithm(graph, reversed); },
                            "Combined sequence/DAG cycle detected");

    sequences = initial_schedule(graph);
    Sequences missing = sequences;
    missing[0].pop_back();
    throws<InvalidSequence>([&] { kernel_algorithm(graph, missing); },
                            "Missing sequence task");

    auto bad = graph;
    bad[0].execution_start_time =
        std::numeric_limits<double>::quiet_NaN();
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "NaN starts rejected safely");

    bad = graph;
    bad[1].execution_start_time = 0.5;
    bad[1].FT_l = 1.5;
    bad[1].execution_finish_time = 1.5;
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "Overlap and dependency corruption rejected");

    bad = graph;
    bad[1].FT_l = 99;
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "Duration mismatch rejected");

    bad = graph;
    bad[1].is_scheduled = SchedulingState::UNSCHEDULED;
    require(!std::get<0>(validate_schedule_constraints(bad)),
            "Incomplete schedules rejected");
}

// Generic resources and randomized independent oracle ----------------------

void generic_resources_and_random_oracle() {
    for (std::size_t cores : {1U, 3U, 6U}) {
        for (unsigned seed = 1; seed <= 35; ++seed) {
            std::mt19937 random(seed * 37 + static_cast<unsigned>(cores));
            const std::size_t task_count = 3 + seed % 6;

            std::vector<int> ids;
            ids.reserve(task_count);
            std::map<int, std::vector<Time>> durations;
            for (std::size_t index = 0; index < task_count; ++index) {
                const int id = static_cast<int>(index * 7 + 10);
                ids.push_back(id);
                auto& local = durations[id];
                local.reserve(cores);
                for (std::size_t core = 0; core < cores; ++core) {
                    local.push_back(1 + random() % 20);
                }
            }

            std::vector<std::pair<int, int>> edges;
            edges.reserve(task_count * (task_count - 1) / 8 + 1);
            for (std::size_t source = 0; source < task_count; ++source) {
                for (std::size_t destination = source + 1;
                     destination < task_count; ++destination) {
                    if (random() % 4 == 0) {
                        edges.emplace_back(ids[source], ids[destination]);
                    }
                }
            }

            // IDs need not track vector positions.
            std::shuffle(ids.begin(), ids.end(), random);
            auto tasks = create_task_graph(ids, durations, {1, 1, 1}, edges);
            for (auto& task : tasks) {
                task.cloud_execution_times = {
                    Time(1 + random() % 4), Time(1 + random() % 10),
                    Time(random() % 8)};
            }

            Sequences sequences = initial_schedule(tasks);
            valid(tasks, sequences);
            matches_oracle(tasks, sequences);

            KernelStatistics statistics;
            auto rescheduled = tasks;
            kernel_algorithm(rescheduled, sequences, &statistics);
            require(statistics.scheduled_tasks == task_count &&
                        statistics.dag_edge_updates == edges.size(),
                    "Kernel visits each DAG edge once");
            valid(rescheduled, sequences);
            matches_oracle(rescheduled, sequences);

            // Check every allowed single-task migration against the oracle.
            for (std::size_t index = 0; index < task_count; ++index) {
                if (tasks[index].assignment == static_cast<int>(cores)) {
                    continue;
                }
                for (std::size_t destination = 0; destination <= cores;
                     ++destination) {
                    if (tasks[index].assignment ==
                        static_cast<int>(destination)) {
                        continue;
                    }

                    auto trial = tasks;
                    Sequences changed = construct_sequence(
                        trial, trial[index].id, static_cast<int>(destination),
                        sequences);
                    bool oracle_cycle = false;
                    try {
                        (void)event_oracle(trial, changed);
                    } catch (const InvalidSequence&) {
                        oracle_cycle = true;
                    }

                    if (oracle_cycle) {
                        throws<InvalidSequence>(
                            [&] { kernel_algorithm(trial, changed); },
                            "Kernel rejects oracle-detected cycle");
                    } else {
                        kernel_algorithm(trial, changed);
                        valid(trial, changed);
                        matches_oracle(trial, changed);
                    }
                }
            }

            const Time deadline = total_time(tasks) + 8.25;
            auto optimized =
                optimize_task_scheduling(tasks, sequences, deadline);
            valid(optimized.first, optimized.second, deadline);
            matches_oracle(optimized.first, optimized.second);

            const auto powers = default_core_powers(cores);
            const double final_energy =
                total_energy(optimized.first, powers, 0.5);
            require(final_energy <= total_energy(tasks, powers, 0.5),
                    "Optimization never increases energy");

            for (std::size_t index = 0; index < task_count; ++index) {
                if (optimized.first[index].assignment ==
                    static_cast<int>(cores)) {
                    continue;
                }
                for (std::size_t destination = 0; destination <= cores;
                     ++destination) {
                    if (optimized.first[index].assignment ==
                        static_cast<int>(destination)) {
                        continue;
                    }

                    auto trial = optimized.first;
                    Sequences changed = construct_sequence(
                        trial, trial[index].id, static_cast<int>(destination),
                        optimized.second);
                    try {
                        kernel_algorithm(trial, changed);
                    } catch (const InvalidSequence&) {
                        continue;
                    }
                    require(
                        total_time(trial) > deadline ||
                            total_energy(trial, powers, 0.5) >= final_energy,
                        "Outer loop terminates only when no improving feasible "
                        "allowed move remains");
                }
            }
        }
    }
}

// Scalability and Section IV experiment helpers -----------------------------

void linear_kernel_and_experiments() {
    constexpr int task_count = 5000;
    std::vector<Task> chain;
    chain.reserve(task_count);
    Sequences sequences(7);
    sequences[0].reserve(task_count);

    for (int id = 1; id <= task_count; ++id) {
        chain.emplace_back(id, std::vector<Time>{1, 2, 3, 4, 5, 6},
                           std::array<Time, 3>{1, 1, 1});
        if (id > 1) {
            chain.back().pred_tasks.push_back(id - 1);
        }
        if (id < task_count) {
            chain.back().succ_tasks.push_back(id + 1);
        }
        chain.back().assignment = 0;
        sequences[0].push_back(id);
    }

    KernelStatistics statistics;
    kernel_algorithm(chain, sequences, &statistics);
    require(statistics.scheduled_tasks == task_count &&
                statistics.dag_edge_updates == task_count - 1 &&
                statistics.sequence_edge_updates == task_count - 1,
            "5000-task kernel performs N task, E DAG-edge, and N-1 "
            "sequence-edge updates");
    near(total_time(chain), task_count, "Long chain completion");
    task_prioritizing(chain);
    near(chain.front().priority_score, 3.5 * task_count,
         "Iterative ranks avoid recursive stack overflow");
    matches_oracle(chain, sequences);

    GeneratorConfig config;
    config.tasks = 9;
    config.cores = 6;
    config.seed = 123;
    auto first = generate_task_graph(config);
    auto second = generate_task_graph(config);
    require(first.size() == 9 && core_count(first) == 6,
            "Generator supports K=6");
    for (std::size_t index = 0; index < first.size(); ++index) {
        require(first[index].core_execution_times ==
                        second[index].core_execution_times &&
                    first[index].cloud_execution_times ==
                        second[index].cloud_execution_times &&
                    first[index].succ_tasks == second[index].succ_tasks,
                "Seeded generator reproducible");
    }

    auto result = random_assignment_baseline(first, 10000, {}, 100, 9);
    require(result.feasible && result.trials == 100,
            "Baseline 1 evaluates requested random assignments");
    valid(result.tasks, result.sequences, 10000);
    matches_oracle(result.tasks, result.sequences);

    auto repeat = random_assignment_baseline(first, 10000, {}, 100, 9);
    const auto six_core_powers = default_core_powers(6);
    near(total_energy(repeat.tasks, six_core_powers, 0.5),
         total_energy(result.tasks, six_core_powers, 0.5),
         "Seeded baseline reproducible");

    auto impossible = random_assignment_baseline(first, 0, {}, 10, 9);
    require(!impossible.feasible,
            "Baseline reports no feasible sample without presenting it as "
            "success");

    auto expensive_rf =
        create_task_graph({1}, {{1, {9, 7, 5}}}, {3, 1, 1}, {});
    auto configured = random_assignment_baseline(
        expensive_rf, 7.5, {{1, 2, 4}, 100}, 100, 9);
    require(configured.feasible,
            "Configured-energy baseline found feasible sample");
    near(total_energy(configured.tasks, {1, 2, 4}, 100), 14,
         "Baseline candidate evaluation uses configured RF power");

    config.edge_density = 0;
    auto disconnected = generate_task_graph(config);
    for (const auto& task : disconnected) {
        require(task.pred_tasks.empty() && task.succ_tasks.empty(),
                "Zero-density multi-entry/multi-exit generator");
    }

    config.edge_density = 1;
    auto dense = generate_task_graph(config);
    std::size_t edge_count = 0;
    for (const auto& task : dense) {
        edge_count += task.succ_tasks.size();
    }
    require(edge_count == config.tasks * (config.tasks - 1) / 2,
            "Unit density creates complete forward DAG");

    config.edge_density = 2;
    throws([&] { generate_task_graph(config); }, "Invalid density rejected");
}

void exhaustive_small_sequences() {
    auto graph = create_task_graph(
        {1, 2, 3}, {{1, {4, 2}}, {2, {3, 5}}, {3, {6, 3}}}, {1, 4, 2},
        {{1, 3}, {2, 3}});
    graph[1].cloud_execution_times = {2, 1, 5};

    // All 3^3 assignments and every within-resource permutation. Compare valid
    // schedules and cycle rejection against a separately constructed event DAG.
    for (int code = 0; code < 27; ++code) {
        int encoded = code;
        auto tasks = graph;
        Sequences sequences(3);
        for (auto& task : tasks) {
            task.assignment = encoded % 3;
            encoded /= 3;
            sequences[static_cast<std::size_t>(task.assignment)].push_back(
                task.id);
        }

        const auto enumerate = [&](const auto& self,
                                   std::size_t resource) -> void {
            if (resource == sequences.size()) {
                bool cycle = false;
                try {
                    (void)event_oracle(tasks, sequences);
                } catch (const InvalidSequence&) {
                    cycle = true;
                }

                auto trial = tasks;
                if (cycle) {
                    throws<InvalidSequence>(
                        [&] { kernel_algorithm(trial, sequences); },
                        "Exhaustive cyclic sequence rejected");
                } else {
                    kernel_algorithm(trial, sequences);
                    valid(trial, sequences);
                    matches_oracle(trial, sequences);
                }
                return;
            }

            auto& sequence = sequences[resource];
            std::sort(sequence.begin(), sequence.end());
            do {
                self(self, resource + 1);
            } while (std::next_permutation(sequence.begin(), sequence.end()));
        };
        enumerate(enumerate, 0);
    }
}

}  // namespace

int main() {
    struct TestCase {
        std::string_view name;
        void (*run)();
    };
    static constexpr std::array<TestCase, 9> tests = {{
        {"paper figures and equations", paper_examples},
        {"parallel cloud and phase dependencies",
         cloud_parallelism_and_dependencies},
        {"priority ordering and earliest gap insertion",
         initial_priority_and_gap_insertion},
        {"migration insertion and two-stage policy",
         migration_insertion_and_policy},
        {"configured energy and explicit hard deadlines",
         configured_energy_and_deadlines},
        {"malformed inputs and schedules", malformed_inputs_and_schedules},
        {"generic resources and independent random event-DAG oracle",
         generic_resources_and_random_oracle},
        {"linear kernel and experimental baselines",
         linear_kernel_and_experiments},
        {"exhaustive small-graph assignments and sequence permutations",
         exhaustive_small_sequences},
    }};

    try {
        for (const TestCase& test : tests) {
            test.run();
            std::cout << "PASS: " << test.name << '\n';
        }
        std::cout << "PASS: " << checks << " checks\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what()
                  << '\n';
        return 1;
    }
}
