#ifndef MCC_SCHEDULER_HPP
#define MCC_SCHEDULER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Time = double;
using Sequences = std::vector<std::vector<int>>;

enum class SchedulingState { UNSCHEDULED, SCHEDULED, KERNEL_SCHEDULED };

// IDs need only be unique, not contiguous. Local resources use indices [0,K),
// cloud uses K (paper notation: local 1..K, cloud 0).
struct TaskSchedule {
    Time FT_l=0, FT_ws=0, FT_c=0, FT_wr=0;
    Time RT_l=0, RT_ws=0, RT_c=0, RT_wr=0;
    double priority_score=0;
    int assignment=-1;
    bool is_core_task=true;
    Time execution_start_time=0;
    Time execution_finish_time=0;
    SchedulingState is_scheduled=SchedulingState::UNSCHEDULED;
};
struct Task : TaskSchedule {
    int id;
    std::vector<int> pred_tasks, succ_tasks;
    std::vector<Time> core_execution_times;
    std::array<Time,3> cloud_execution_times;
    Task(int task_id, std::vector<Time> local_times, std::array<Time,3> remote_times);
};

struct EnergyModel {
    std::vector<double> core_powers;
    double sending_power=0.5;
};

struct KernelStatistics {
    std::size_t scheduled_tasks=0;
    std::size_t dag_edge_updates=0;
    std::size_t sequence_edge_updates=0;
};

struct MigrationStatistics {
    std::size_t candidates=0;
    std::size_t rejected_sequences=0;
    std::size_t accepted=0;
};

class InvalidSequence : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class DeadlineNotMet : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ScheduleResult {
    std::vector<Task> initial_tasks, tasks;
    Sequences initial_sequences, sequences;
    Time deadline=0;
    bool feasible=false;
    MigrationStatistics migrations;
};

// Reject invalid durations, inconsistent edges, duplicates, and cycles.
void validate_task_graph(const std::vector<Task>& tasks);
std::size_t core_count(const std::vector<Task>& tasks);
std::vector<Task> create_task_graph(
    const std::vector<int>& ids,
    const std::map<int,std::vector<Time>>& local_times,
    const std::array<Time,3>& cloud_times,
    const std::vector<std::pair<int,int>>& edges);

Time total_time(const std::vector<Task>& tasks);
// Equations 1 and 2: consistent data/rate units produce a duration.
Time transmission_time(double data_amount, double data_rate);
double calculate_energy_consumption(const Task& task, const std::vector<double>& powers, double sending_power);
double total_energy(const std::vector<Task>& tasks, const std::vector<double>& powers, double sending_power);
std::vector<double> default_core_powers(std::size_t count);

// Paper III.A: primary classification, upward ranks, earliest-slot selection.
void primary_assignment(std::vector<Task>& tasks, std::size_t expected_cores=0, bool cloud_enabled=true);
void task_prioritizing(std::vector<Task>& tasks);
Sequences execution_unit_selection(std::vector<Task>& tasks, bool cloud_enabled=true);
Sequences initial_schedule(std::vector<Task>& tasks, bool cloud_enabled=true);
// Baseline 1: assignments predefined, otherwise the same rank/slot scheduler.
Sequences fixed_assignment_schedule(std::vector<Task>& tasks, const std::vector<int>& assignments);

// Equation 19 uses destination ready time computed from the OLD schedule.
Sequences construct_sequence(std::vector<Task>& tasks, int task_id, int destination, Sequences sequences);
// LIFO, incremental dependency and sequence updates: O(N+E+K).
std::vector<Task>& kernel_algorithm(std::vector<Task>& tasks, const Sequences& sequences,
                                   KernelStatistics* statistics=nullptr);
// Deadline is the actual T_max, NOT an initial-time multiplier.
std::pair<std::vector<Task>,Sequences> optimize_task_scheduling(
    std::vector<Task> tasks, Sequences sequences, Time deadline,
    std::vector<double> core_powers={}, double sending_power=0.5,
    bool cloud_enabled=true, MigrationStatistics* statistics=nullptr);
ScheduleResult schedule_application(std::vector<Task> tasks, Time deadline,
                                    EnergyModel energy={}, bool cloud_enabled=true);

// Independent interval/equation checks. Cloud computation and downloads may
// overlap. Only local-core and upload conflicts are invalid in the paper model.
std::tuple<bool,std::vector<std::string>> validate_schedule_constraints(
    const std::vector<Task>& tasks, Time deadline=std::numeric_limits<Time>::infinity(),
    const Sequences* sequences=nullptr);
void print_schedule_tasks(const std::vector<Task>& tasks);
void print_schedule_sequences(const Sequences& sequences);
void print_schedule_validation_report(const std::vector<Task>& tasks, Time deadline=std::numeric_limits<Time>::infinity());

std::vector<std::vector<Task>> example_graphs();

// Section IV methodology. Distributions/seeds were not published; our explicit
// uniform sampling choices reproduce the methodology, not the original tables.
struct GeneratorConfig {
    std::size_t tasks=11, cores=3;
    double edge_density=0.15;
    double local_mean=20, send_mean=3, compute_mean=1, receive_mean=1;
    double speedup=1.5;
    std::uint64_t seed=1;
};
std::vector<Task> generate_task_graph(const GeneratorConfig& config);
struct BaselineResult {
    std::vector<Task> tasks;
    Sequences sequences;
    bool feasible=false;
    std::size_t trials=0;
};
BaselineResult random_assignment_baseline(const std::vector<Task>& graph, Time deadline,
    EnergyModel energy={}, std::size_t trials=10000, std::uint64_t seed=1);

#endif
