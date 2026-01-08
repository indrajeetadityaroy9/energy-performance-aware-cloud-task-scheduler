#ifndef MCC_SCHEDULER_HPP
#define MCC_SCHEDULER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/**
 * @brief Tracks the scheduling progress of each task through different phases
 */
enum class SchedulingState {
    UNSCHEDULED = 0,      // Initial state before scheduling
    SCHEDULED = 1,        // Task has been scheduled in initial phase
    KERNEL_SCHEDULED = 2  // Task has been processed by kernel algorithm during migration
};

/**
 * @brief Represents a node in the directed acyclic task graph (DAG)
 *
 * Each task has execution time requirements for different processing units,
 * maintains precedence relationships, and tracks scheduling state.
 */
class Task {
public:
    // Task identification and graph structure
    int id;  // Unique task identifier
    std::vector<int> pred_tasks;  // Immediate predecessors in the task graph
    std::vector<int> succ_tasks;  // Immediate successors in the task graph

    // Execution times for different processing units
    std::array<int,3> core_execution_times;  // T_i^l for each local core k
    std::array<int,3> cloud_execution_times; // [send, compute, receive] times

    // Finish times for different execution phases
    int FT_l;   // Local core finish time
    int FT_ws;  // Wireless sending finish time
    int FT_c;   // Cloud computation finish time
    int FT_wr;  // Wireless receiving finish time

    // Ready times for different execution phases
    int RT_l;   // Ready time for local execution (Equation 3)
    int RT_ws;  // Ready time for wireless sending (Equation 4)
    int RT_c;   // Ready time for cloud execution (Equation 5)
    int RT_wr;  // Ready time for receiving results (Equation 6)

    // Scheduling metadata
    double priority_score;  // Priority score used in initial scheduling (Equations 15-16)
    int assignment;         // Task assignment: -1 for cloud, 0...K-1 for local cores
    bool is_core_task;      // Flags whether task is assigned to core (vs cloud)

    std::vector<int> execution_unit_task_start_times;  // Start times for task on different units
    int execution_finish_time;  // Overall finish time
    SchedulingState is_scheduled;  // Current state in scheduling process

    /**
     * @brief Constructs a task with its execution time requirements
     *
     * @param task_id Unique identifier for this task
     * @param core_exec_times_map Execution times for each core
     * @param cloud_exec_times_input Cloud execution phase times [send, compute, receive]
     */
    Task(int task_id,
         const std::map<int, std::array<int,3>>& core_exec_times_map,
         const std::array<int,3>& cloud_exec_times_input);
};

/**
 * @brief Implements the initial scheduling algorithm for minimal-delay scheduling
 *
 * This class handles Phase 1 of the algorithm, generating an initial schedule
 * that minimizes application completion time.
 */
class InitialTaskScheduler {
public:
    /**
     * @brief Constructs the scheduler with tasks and available resources
     *
     * @param tasks Reference to tasks to be scheduled
     * @param num_cores Number of local cores available (default: 3)
     */
    InitialTaskScheduler(std::vector<Task>& tasks, int num_cores=3);

    /**
     * @brief Represents a potential core assignment for a task
     */
    struct CoreChoice {
        int core;         // Which core would be used
        int start_time;   // When the task could start
        int finish_time;  // When the task would complete
    };

    /**
     * @brief Implements the timing structure for cloud execution
     */
    struct CloudTiming {
        int send_ready;      // RT_ws: Ready time for wireless sending
        int send_finish;     // FT_ws: When data transmission completes
        int cloud_ready;     // RT_c: When cloud can begin processing
        int cloud_finish;    // FT_c: When cloud computation completes
        int receive_ready;   // RT_wr: When results can begin downloading
        int receive_finish;  // FT_wr: When mobile device has complete results
    };

    /**
     * @brief Returns tasks ordered by descending priority
     */
    std::vector<int> get_priority_ordered_tasks();

    /**
     * @brief Separates tasks into entry tasks and non-entry tasks
     *
     * @param priority_order Tasks in priority order
     * @return Pair of (entry_tasks, non_entry_tasks)
     */
    std::pair<std::vector<Task*>, std::vector<Task*>> classify_entry_tasks(
        const std::vector<int>& priority_order
    );

    /**
     * @brief Finds the best local core for a task by minimizing completion time
     *
     * @param task Task to schedule
     * @param ready_time Earliest time task can start due to dependencies
     * @return CoreChoice with optimal core assignment
     */
    CoreChoice identify_optimal_local_core(Task &task, int ready_time=0);

    /**
     * @brief Schedules a task on a local core
     *
     * @param task Task being scheduled
     * @param core Target core for execution
     * @param start_time When task will begin
     * @param finish_time When task will complete
     */
    void schedule_on_local_core(Task &task, int core, int start_time, int finish_time);

    /**
     * @brief Calculates timing for all three phases of cloud execution
     *
     * @param task Task to schedule
     * @return CloudTiming with complete timeline
     */
    CloudTiming calculate_cloud_phases_timing(Task &task);

    /**
     * @brief Schedules a task for cloud execution
     *
     * @param task Task being scheduled
     * @param send_ready When can start sending data
     * @param send_finish When data transmission completes
     * @param cloud_ready When cloud computation can begin
     * @param cloud_finish When cloud computation ends
     * @param receive_ready When can start receiving results
     * @param receive_finish When all results are received
     */
    void schedule_on_cloud(Task &task, int send_ready, int send_finish,
                          int cloud_ready, int cloud_finish,
                          int receive_ready, int receive_finish);

    /**
     * @brief Schedules all entry tasks (tasks with no predecessors)
     *
     * @param entry_tasks Vector of entry task pointers
     */
    void schedule_entry_tasks(std::vector<Task*> &entry_tasks);

    /**
     * @brief Calculates ready times for non-entry tasks based on predecessors
     *
     * @param task Task to calculate ready times for
     */
    void calculate_non_entry_task_ready_times(Task &task);

    /**
     * @brief Schedules non-entry tasks with dependency and resource constraints
     *
     * @param non_entry_tasks Vector of non-entry task pointers
     */
    void schedule_non_entry_tasks(std::vector<Task*> &non_entry_tasks);

    /**
     * @brief Returns the final scheduling sequences for all execution units
     *
     * @return Vector of sequences, one per execution unit
     */
    std::vector<std::vector<int>> get_sequences() const;

private:
    std::vector<Task> &tasks;                // All tasks in the task graph
    int k;                                   // Number of available local cores
    std::vector<int> core_earliest_ready;    // Tracks when each core will be available
    int ws_ready;                            // Tracks when wireless sending channel is free
    int wr_ready;                            // Tracks when wireless receiving channel is free
    std::vector<std::vector<int>> sequences; // Execution sequences for each resource
};

/**
 * @brief Implements the linear-time kernel algorithm for rescheduling
 *
 * This class handles rescheduling after task migrations while maintaining
 * all precedence constraints.
 */
class KernelScheduler {
public:
    /**
     * @brief Constructs the scheduler with tasks and current sequences
     *
     * @param tasks Reference to all tasks
     * @param sequences Reference to current execution sequences
     */
    KernelScheduler(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences);

    /**
     * @brief Sets up initial task state tracking for dependencies and sequences
     *
     * @return Pair of (dependency_ready, sequence_ready) vectors
     */
    std::pair<std::vector<int>, std::vector<int>> initialize_task_state();

    /**
     * @brief Updates the readiness state of a task
     *
     * @param task Task to update
     */
    void update_task_state(Task &task);

    /**
     * @brief Schedules a task for execution on a local core
     *
     * @param task Task to schedule
     */
    void schedule_local_task(Task &task);

    /**
     * @brief Schedules a task for cloud execution
     *
     * @param task Task to schedule
     */
    void schedule_cloud_task(Task &task);

    /**
     * @brief Initializes the scheduling queue with ready tasks
     *
     * @return Queue of tasks ready for scheduling
     */
    std::deque<Task*> initialize_queue();

    // Public member variables for state tracking
    std::vector<Task> &tasks;                      // Task graph tasks
    std::vector<std::vector<int>> &sequences;      // Current execution sequences
    std::array<int,3> RT_ls;                       // Ready times for local cores
    std::array<int,3> cloud_phases_ready_times;    // Ready times for cloud phases
    std::vector<int> dependency_ready;             // Dependency counters per task
    std::vector<int> sequence_ready;               // Sequence readiness per task
};

/**
 * @brief Defines a unique key for caching migration scenarios
 */
struct MigrationKey {
    int task_idx;                     // Which task is being moved
    int target_execution_unit;        // Target location
    std::vector<int> assignments;     // Current assignments of all tasks

    /**
     * @brief Comparison operator for use in std::map
     */
    bool operator<(const MigrationKey& other) const;
};

/**
 * @brief Represents the complete state of a task migration decision
 */
struct TaskMigrationState {
    int time;                    // Total completion time after migration
    double energy;               // Total energy consumption after migration
    double efficiency;           // Energy savings per unit time increase
    int task_index;              // Which task to migrate
    int target_execution_unit;   // Where to migrate the task
};

/**
 * @brief Represents a potential migration candidate
 */
struct MigrationCandidate {
    double neg_efficiency;       // Negative efficiency for priority queue ordering
    int task_idx;                // Task to potentially migrate
    int resource_idx;            // Target resource for migration
    int time;                    // Resulting completion time
    double energy;               // Resulting energy consumption

    /**
     * @brief Comparison operator for priority queue (higher efficiency = higher priority)
     */
    bool operator<(const MigrationCandidate& other) const;
};

/**
 * @brief Calculates the total application completion time
 *
 * Implements Equation 10: T^total = max(max(FT_i^l, FT_i^wr)) for all exit tasks
 *
 * @param tasks All tasks in the application
 * @return Total completion time
 */
int total_time(const std::vector<Task>& tasks);

/**
 * @brief Calculates the energy consumption for a single task
 *
 * Uses Equation 7 for local execution or Equation 8 for cloud execution
 *
 * @param task Task being evaluated
 * @param core_powers Power consumption values for each core
 * @param cloud_sending_power RF sending power for cloud communication
 * @return Energy consumption for the task
 */
double calculate_energy_consumption(
    const Task& task,
    const std::vector<int>& core_powers,
    double cloud_sending_power
);

/**
 * @brief Calculates the total energy consumption for the entire application
 *
 * Implements Equation 9: E^total = Σ(E_i) for i=1 to N
 *
 * @param tasks All tasks in the application
 * @param core_powers Power consumption values for each core
 * @param cloud_sending_power Power consumption for RF transmission
 * @return Total energy consumption
 */
double total_energy(
    const std::vector<Task>& tasks,
    const std::vector<int>& core_powers,
    double cloud_sending_power
);

/**
 * @brief Implements the primary assignment phase
 *
 * Makes initial decisions about which tasks should be considered for cloud execution
 * based on comparing minimum local execution time with remote execution time.
 *
 * @param tasks All tasks in the application graph
 * @param k Number of cores available
 */
void primary_assignment(std::vector<Task>& tasks, int k);

/**
 * @brief Recursively calculates task priorities
 *
 * Implements Equations 15 and 16 for priority calculation
 *
 * @param task Current task
 * @param tasks All tasks
 * @param w Computation costs
 * @param computed_priority_scores Memoization cache
 * @return Priority score for the task
 */
double calculate_priority(
    const Task& task,
    const std::vector<Task>& tasks,
    const std::vector<double>& w,
    std::map<int,double>& computed_priority_scores
);

/**
 * @brief Implements the task prioritizing phase
 *
 * Calculates priority levels for all tasks to determine scheduling order
 *
 * @param tasks All tasks to prioritize
 */
void task_prioritizing(std::vector<Task>& tasks);

/**
 * @brief Implements the complete execution unit selection phase
 *
 * Generates initial task schedule that minimizes completion time
 *
 * @param tasks All tasks to schedule
 * @return Execution sequences for each resource
 */
std::vector<std::vector<int>> execution_unit_selection(std::vector<Task>& tasks);

/**
 * @brief Constructs a new sequence when migrating a task
 *
 * @param tasks All tasks in the system
 * @param task_id Task being migrated
 * @param execution_unit New execution unit for the task
 * @param original_sequence Current scheduling
 * @return Updated sequence with migrated task
 */
std::vector<std::vector<int>> construct_sequence(
    std::vector<Task>& tasks,
    int task_id,
    int execution_unit,
    std::vector<std::vector<int>> original_sequence
);

/**
 * @brief Implements the linear-time kernel algorithm
 *
 * Reschedules tasks after migrations while maintaining dependencies
 *
 * @param tasks All tasks to reschedule
 * @param sequences Current execution sequences
 * @return Reference to updated tasks
 */
std::vector<Task>& kernel_algorithm(
    std::vector<Task>& tasks,
    std::vector<std::vector<int>>& sequences
);

/**
 * @brief Creates a cache key for a specific migration scenario
 *
 * @param tasks All tasks in the system
 * @param task_idx Task being migrated
 * @param target_execution_unit Destination for the task
 * @return MigrationKey capturing the complete state
 */
MigrationKey generate_cache_key(
    const std::vector<Task>& tasks,
    int task_idx,
    int target_execution_unit
);

/**
 * @brief Evaluates the impact of moving a task to a new execution unit
 *
 * @param tasks All tasks in system
 * @param seqs Current execution sequences
 * @param task_idx Task considering moving
 * @param target_execution_unit Target location
 * @param migration_cache Cache of previous evaluations
 * @param core_powers Power consumption of cores
 * @param cloud_sending_power Power for cloud communication
 * @return Pair of (completion_time, energy_consumption)
 */
std::pair<int,double> evaluate_migration(
    std::vector<Task>& tasks,
    const std::vector<std::vector<int>>& seqs,
    int task_idx,
    int target_execution_unit,
    std::map<MigrationKey, std::pair<int,double>>& migration_cache,
    const std::vector<int>& core_powers = {1,2,4},
    double cloud_sending_power = 0.5
);

/**
 * @brief Initializes a matrix of valid migration choices for each task
 *
 * @param tasks All tasks in the system
 * @return Matrix of valid migration destinations per task
 */
std::vector<std::array<bool,4>> initialize_migration_choices(
    const std::vector<Task>& tasks
);

/**
 * @brief Identifies the optimal migration among candidates
 *
 * @param migration_trials_results Results from all migration evaluations
 * @param T_final Current completion time
 * @param E_total Current energy consumption
 * @param T_max Maximum allowed completion time
 * @return Pointer to best migration state (caller must delete), or nullptr if none found
 */
TaskMigrationState* identify_optimal_migration(
    const std::vector<std::tuple<int,int,int,double>>& migration_trials_results,
    int T_final,
    double E_total,
    int T_max
);

/**
 * @brief Implements the task scheduling optimization
 *
 * Iteratively improves energy consumption through strategic task migrations
 *
 * @param tasks Tasks to be scheduled
 * @param sequence Initial task sequences
 * @param T_final Target completion time
 * @param core_powers Power consumption of cores
 * @param cloud_sending_power Power for cloud communication
 * @return Pair of (optimized_tasks, optimized_sequences)
 */
std::pair<std::vector<Task>, std::vector<std::vector<int>>> optimize_task_scheduling(
    std::vector<Task> tasks,
    std::vector<std::vector<int>> sequence,
    int T_final,
    std::vector<int> core_powers = {1, 2, 4},
    double cloud_sending_power = 0.5
);

/**
 * @brief Prints the task schedule in tabular format
 *
 * @param tasks All tasks to display
 */
void print_schedule_tasks(const std::vector<Task>& tasks);

/**
 * @brief Validates schedule constraints and returns violations
 *
 * Checks 5 critical constraints:
 * 1. Wireless sending channel conflicts
 * 2. Cloud computing conflicts
 * 3. Wireless receiving channel conflicts
 * 4. Task precedence dependencies
 * 5. Core execution conflicts
 *
 * @param tasks All tasks to validate
 * @return Tuple of (is_valid, violations_list)
 */
std::tuple<bool, std::vector<std::string>> validate_schedule_constraints(
    const std::vector<Task>& tasks
);

/**
 * @brief Prints the schedule validation report
 *
 * @param tasks All tasks to validate
 */
void print_schedule_validation_report(const std::vector<Task>& tasks);

/**
 * @brief Prints the execution sequences for all resources
 *
 * @param sequences Execution sequences to display
 */
void print_schedule_sequences(const std::vector<std::vector<int>>& sequences);

/**
 * @brief Creates a complete task graph with execution times and dependencies
 *
 * @param task_ids Unique IDs for each task
 * @param core_exec_times Execution times on local cores
 * @param cloud_exec_times Cloud execution phase times [send, compute, receive]
 * @param edges Task dependencies as (predecessor, successor) pairs
 * @return Vector of constructed tasks forming a DAG
 */
std::vector<Task> create_task_graph(
    const std::vector<int>& task_ids,
    const std::map<int, std::array<int,3>>& core_exec_times,
    const std::array<int,3>& cloud_exec_times,
    const std::vector<std::pair<int,int>>& edges
);

#endif // MCC_SCHEDULER_HPP
