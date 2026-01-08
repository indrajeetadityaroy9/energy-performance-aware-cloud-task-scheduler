#include "mcc.hpp"

using namespace std;

// Constructor initializes a task with its execution time requirements
Task::Task(int task_id,
           const map<int, array<int,3>>& core_exec_times_map,
           const array<int,3>& cloud_exec_times_input)
    : id(task_id),
      FT_l(0),
      FT_ws(0),
      FT_c(0),
      FT_wr(0),
      RT_l(-1),
      RT_ws(-1),
      RT_c(-1),
      RT_wr(-1),
      priority_score(-1.0),
      assignment(-2),
      is_core_task(false),
      execution_finish_time(-1),
      is_scheduled(SchedulingState::UNSCHEDULED)
{
    // Set local core execution times if available
    auto it = core_exec_times_map.find(id);
    if (it != core_exec_times_map.end()) {
        this->core_execution_times = it->second;
    } else {
        this->core_execution_times = {0,0,0};
    }

    // Set cloud execution phase times
    this->cloud_execution_times = cloud_exec_times_input;
}

// Calculates the total application completion time
// T^total = max(max(FT_i^l, FT_i^wr)) for all exit tasks
int total_time(const vector<Task>& tasks) {
    // Track the maximum completion time across all exit tasks
    int max_completion_time = 0;
    // Iterate through all tasks to find exit tasks (tasks with no successors)
    for (const auto& task : tasks) {
        // Check if this is an exit task
        // An exit task has no successors (succ_tasks is empty)
        if (task.succ_tasks.empty()) {
            // For each exit task, find its actual completion time by taking
            // the maximum of:
            // - FT_l: finish time if executed locally on a core
            // - FT_wr: finish time if offloaded to cloud (when results are received)
            // This implements the inner max() in Equation 10
            int completion = max(task.FT_l, task.FT_wr);
            // Keep track of the maximum completion time across all exit tasks
            // This implements the outer max() in Equation 10
            if (completion > max_completion_time) {
                max_completion_time = completion;
            }
        }
    }
    // Return T^total - the application completion time
    return max_completion_time;
}

// Calculates the energy consumption for a single task
// Uses either Equation 7 (for local execution) or Equation 8 (for cloud execution)
double calculate_energy_consumption(
    const Task& task,                          // Task being evaluated
    const vector<int>& core_powers,       // P_k values for each core
    double cloud_sending_power                 // P^s value for RF sending power
) {
    // Check if task is assigned to a local core
    if (task.is_core_task) {
        // Get the core this task is assigned to
        int core_index = task.assignment;
        // Implement Equation 7: E_i^l = P_k * T_i^l
        // Where:
        // - P_k is the power consumption of core k (from core_powers)
        // - T_i^l is the execution time of task i on core k (from core_execution_times)
        return static_cast<double>(core_powers[core_index]) * static_cast<double>(task.core_execution_times[core_index]);

    } else {
        // Implement Equation 8: E_i^c = P^s * T_i^s
        // Where:
        // - P^s is the RF sending power consumption (cloud_sending_power)
        // - T_i^s is the time to send task i to cloud (cloud_execution_times[0])
        return static_cast<double>(cloud_sending_power) * static_cast<double>(task.cloud_execution_times[0]);
    }
}

// Calculates the total energy consumption E^total for the entire application as defined in Equation 9
double total_energy(
    const vector<Task>& tasks,            // All tasks in the application
    const vector<int>& core_powers,       // Power consumption values for each core
    double cloud_sending_power                 // Power consumption for RF transmission
) {
    // Initialize the total energy consumption
    double total = 0.0;
    // Implement Equation 9: E^total = Σ(E_i) for i=1 to N
    // where E_i is either E_i^l (local execution) or E_i^c (cloud execution)
    // depending on task assignment
    for (const auto& task : tasks) {
        // Calculate and accumulate energy consumption for each task
        // using either Equation 7 or 8 depending on execution location
        total += calculate_energy_consumption(task, core_powers, cloud_sending_power);
    }
    // Return the total energy consumption E^total
    return total;
}

// Implements the primary assignment phase
// This phase makes initial decisions about which tasks should be considered for cloud execution
void primary_assignment(
    vector<Task>& tasks,  // All tasks in the application graph
    int k                      // Number of cores available (K)
) {
    // Examine each task to determine if it's better suited for cloud execution
    for (auto& task : tasks) {
        // Calculate T_i^l_min (Equation 11) - the minimum local execution time
        // This represents the best-case scenario for local execution
        // by finding the fastest available core for this task
        int t_l_min = *min_element(task.core_execution_times.begin(), task.core_execution_times.end());

        // Calculate T_i^re (Equation 12) - the remote execution time
        // This is the sum of:
        // - T_i^s: time to send task to cloud (cloud_execution_times[0])
        // - T_i^c: time for cloud computation (cloud_execution_times[1])
        // - T_i^r: time to receive results (cloud_execution_times[2])
        int t_re = task.cloud_execution_times[0] + task.cloud_execution_times[1] + task.cloud_execution_times[2];

        // Compare remote execution time with best local execution time
        // If remote execution is faster, mark the task as a "cloud task"
        if (t_re < t_l_min) {
            task.is_core_task = false;  // Mark for cloud execution
            task.assignment = k;         // k represents cloud assignment
        } else {
            task.is_core_task = true;   // Mark for local execution
        }
    }
}

double calculate_priority(const Task& task, const vector<Task>& tasks, const vector<double>& w, map<int,double>& computed_priority_scores);

// Implements the task prioritizing phase
// This phase calculates priority levels for all tasks to determine scheduling order
void task_prioritizing(vector<Task>& tasks) {
    //Calculate the computation cost (w_i) for each task as described in Equations 13 and 14
    vector<double> w(tasks.size(), 0.0);

    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        if (!task.is_core_task) {
            // For cloud tasks, use Equation 13:
            // w_i = T_i^re (total remote execution time)
            w[i] = static_cast<double>(
                task.cloud_execution_times[0] +  // Send time
                task.cloud_execution_times[1] +  // Cloud computation
                task.cloud_execution_times[2]    // Receive time
            );
        } else {
            // For local tasks, use Equation 14:
            // w_i = average computation time across all cores
            double sum_local = static_cast<double>(accumulate(task.core_execution_times.begin(), task.core_execution_times.end(), 0));
            w[i] = sum_local / static_cast<double>(task.core_execution_times.size());
        }
    }

    // Use memoization to avoid recalculating priorities for the same tasks
    map<int,double> computed_priority_scores;

    // Calculate priorities for all tasks recursively
    for (auto& task : tasks) {
        calculate_priority(task, tasks, w, computed_priority_scores);
    }

    // Assign the computed priorities to each task
    for (auto& task : tasks) {
        task.priority_score = computed_priority_scores[task.id];
    }
}


// Recursively calculates task priorities according to Equations 15 and 16
double calculate_priority(
    const Task& task,                          // Current task
    const vector<Task>& tasks,            // All tasks
    const vector<double>& w,              // Computation costs
    map<int,double>& computed_priority_scores  // Memoization cache
) {
    // Check if priority was already computed
    auto it = computed_priority_scores.find(task.id);
    if (it != computed_priority_scores.end()) {
        return it->second;
    }

    // Base case: Exit tasks (Equation 16)
    // priority(v_i) = w_i for v_i ∈ exit tasks
    if (task.succ_tasks.empty()) {
        double priority_val = w[task.id - 1];
        computed_priority_scores[task.id] = priority_val;
        return priority_val;
    }

    // Recursive case: Non-exit tasks (Equation 15)
    // priority(v_i) = w_i + max(priority(v_j)) for all successors v_j
    double max_successor_priority = -numeric_limits<double>::infinity();
    for (int succ_id : task.succ_tasks) {
        const Task& succ_tasks = tasks[succ_id - 1];
        double succ_priority = calculate_priority(succ_tasks, tasks, w, computed_priority_scores);
        if (succ_priority > max_successor_priority) {
            max_successor_priority = succ_priority;
        }
    }

    // Combine task's computation cost with maximum successor priority
    double task_priority = w[task.id - 1] + max_successor_priority;
    computed_priority_scores[task.id] = task_priority;
    return task_priority;
}

// Constructor initializes the scheduler with tasks and available resources
InitialTaskScheduler::InitialTaskScheduler(vector<Task>& tasks, int num_cores)
    : tasks(tasks),           // Tasks to be scheduled
      k(num_cores),           // Number of local cores available
      ws_ready(0),            // Tracks when wireless sending channel is available
      wr_ready(0)            // Tracks when wireless receiving channel is available
{
    // Initialize arrays to track resource availability
    // core_earliest_ready[i] tracks when core i will next be available
    core_earliest_ready.resize(k, 0);

    // sequences stores the ordered list of tasks for each execution unit
    // Index 0 to k-1 are for local cores
    // Index k is for cloud tasks (wireless sending channel)
    sequences.resize(k+1);
}

// Implements task ordering based on priorities calculated
// Returns tasks sorted by descending priority to determine scheduling order
vector<int> InitialTaskScheduler::get_priority_ordered_tasks() {
        // Create pairs of (priority_score, task_id) for sorting
        vector<pair<double,int>> task_priority_list;
        for (auto &t : tasks) {
            task_priority_list.emplace_back(t.priority_score, t.id);
        }

        // Sort tasks by priority (higher priority first)
        sort(task_priority_list.begin(), 
                 task_priority_list.end(), 
                 [](const auto &a, const auto &b){
                     return a.first > b.first;
                 });

        // Extract just the task IDs in priority order
        vector<int> result;
        for (auto &item : task_priority_list) {
            result.push_back(item.second);
        }
        return result;
    }

// Separates tasks into entry tasks and non-entry tasks while preserving priority order
pair<vector<Task*>, vector<Task*>> InitialTaskScheduler::classify_entry_tasks(
    const vector<int>& priority_order
) {
    vector<Task*> entry_tasks;     // Tasks with no predecessors
    vector<Task*> non_entry_tasks; // Tasks with predecessors

    // Process tasks in priority order, maintaining the scheduling sequence
    for (int task_id : priority_order) {
        Task &task = tasks[task_id - 1];
        // Entry tasks are identified by having no predecessor tasks
        // These can start execution immediately
        if (task.pred_tasks.empty()) {
            entry_tasks.push_back(&task);
        } else {
            non_entry_tasks.push_back(&task);
        }
    }
    return {entry_tasks, non_entry_tasks};
}

// Finds the best local core for a task by minimizing completion time
InitialTaskScheduler::CoreChoice InitialTaskScheduler::identify_optimal_local_core(
    Task &task,
    int ready_time  // Earliest time task can start due to dependencies
) {
    // Initialize with worst-case values to ensure any valid choice is better
    int best_finish_time = numeric_limits<int>::max();
    int best_core = -1;
    int best_start_time = numeric_limits<int>::max();

    // Evaluate each potential core assignment
    for (int core = 0; core < k; core++) {
        // Calculate when the task could actually start on this core
        // Must consider both:
        // - ready_time: when task's dependencies are satisfied
        // - core_earliest_ready[core]: when the core becomes available
        int start_time = max(ready_time, core_earliest_ready[core]);
        
        // Calculate when task would finish on this core
        int finish_time = start_time + task.core_execution_times[core];

        // Update best choice if this core offers earlier completion
        // This directly implements the paper's goal of minimizing finish time
        if (finish_time < best_finish_time) {
            best_finish_time = finish_time;
            best_core = core;
            best_start_time = start_time;
        }
    }

    return {best_core, best_start_time, best_finish_time};
}

// Implements the local task scheduling mechanism
// Handles all aspects of assigning a task to a specific local core
void InitialTaskScheduler::schedule_on_local_core(
    Task &task,          // Task being scheduled
    int core,            // Target core for execution
    int start_time,      // When task will begin
    int finish_time      // When task will complete
) {
    // Record the local finish time (FT_l) as described in Section II.C
    // This is crucial for calculating total completion time
    task.FT_l = finish_time;
    task.execution_finish_time = finish_time;

    // Initialize start times for all execution units
    // Size k+1 accounts for k cores plus 1 cloud channel
    // -1 indicates the task doesn't execute on that unit
    task.execution_unit_task_start_times.assign(k+1, -1);
    // Record when this task starts on its assigned core
    task.execution_unit_task_start_times[core] = start_time;

    // Update core availability for future scheduling decisions
    core_earliest_ready[core] = finish_time;
    // Record final core assignment for this task
    task.assignment = core;
    // Mark task as scheduled in the initial scheduling phase
    task.is_scheduled = SchedulingState::SCHEDULED;
    // Add task to the sequence for this core
    // This maintains the execution order needed for the scheduling algorithm
    sequences[core].push_back(task.id);
}

// Calculates timing for all three phases of cloud execution
// Returns a complete timeline for sending, cloud computation, and receiving results
InitialTaskScheduler::CloudTiming InitialTaskScheduler::calculate_cloud_phases_timing(Task &task) {
        // Phase 1: Wireless Sending Phase
        // RT_ws (send_ready) comes from task dependencies (Equation 4)
        // This represents the earliest time can start sending data to cloud
        int send_ready = task.RT_ws;

        // Calculate when sending phase completes (FT_ws)
        // T_i^s from Equation 1: time to send task data = data_i/R^s
        int send_finish = send_ready + task.cloud_execution_times[0];

        // Phase 2: Cloud Computing Phase
        // Cloud can begin as soon as data transfer completes (Equation 5)
        // This implements the dependency between sending and computing
        int cloud_ready = send_finish;

        // Calculate when cloud computation completes
        // Uses T_i^c: the task's execution time in cloud
        int cloud_finish = cloud_ready + task.cloud_execution_times[1];

        // Phase 3: Results Receiving Phase
        // Can start receiving as soon as cloud computation finishes (Equation 6)
        int receive_ready = cloud_finish;

        // Calculate when results are fully received
        // Must consider both:
        // - When receiving channel (wr_ready) becomes available
        // - When results are ready from cloud (receive_ready)
        // T_i^r from Equation 2: time to receive results = data'_i/R^r
        int receive_finish = max(wr_ready, receive_ready) + task.cloud_execution_times[2];

        // Return complete timeline of all cloud execution phases
        return {send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish};
}

// Records the complete timeline of a task's cloud execution phases
// Implements the cloud scheduling model and timing tracking
void InitialTaskScheduler::schedule_on_cloud(
    Task &task,          // Task being scheduled for cloud execution
    int send_ready,      // RT_ws: When can start sending data
    int send_finish,     // FT_ws: When data transmission completes
    int cloud_ready,     // RT_c: When cloud computation can begin
    int cloud_finish,    // FT_c: When cloud computation ends
    int receive_ready,   // RT_wr: When can start receiving results
    int receive_finish   // FT_wr: When all results are received
) {
        // Record sending phase timing
        // RT_ws comes from task dependencies (Equation 4)
        // FT_ws represents when data transmission completes
        task.RT_ws = send_ready;
        task.FT_ws = send_finish;

        // Record cloud computation phase timing
        // RT_c represents when cloud can begin (Equation 5)
        // FT_c tracks when cloud computation finishes
        task.RT_c = cloud_ready;
        task.FT_c = cloud_finish;

        // Record receiving phase timing
        // RT_wr is when results are ready (Equation 6)
        // FT_wr represents when mobile device has all results
        task.RT_wr = receive_ready;
        task.FT_wr = receive_finish;

        // Update overall task completion tracking
        // For cloud tasks, completion is when results are fully received
        task.execution_finish_time = receive_finish;
        // Set FT_l to 0 since task doesn't execute locally
        task.FT_l = 0;

        // Record task's start time on the wireless sending channel
        // Initialize all execution units to -1 (not used)
        task.execution_unit_task_start_times.assign(k+1, -1);
        // Mark start time on sending channel (index k represents cloud)
        task.execution_unit_task_start_times[k] = send_ready;

        // Mark task as assigned to cloud (represented by index k)
        task.assignment = k;

        // Update task's scheduling state
        task.is_scheduled = SchedulingState::SCHEDULED;

        // Update resource availability
        // Wireless sending channel is busy until send_finish
        ws_ready = send_finish;
        // Wireless receiving channel is busy until receive_finish
        wr_ready = receive_finish;

        // Add task to cloud execution sequence
        sequences[k].push_back(task.id);
}

// Schedules all entry tasks (tasks with no predecessors)
// Handles both local and cloud execution while respecting resource constraints
void InitialTaskScheduler::schedule_entry_tasks(vector<Task*> &entry_tasks) {
        // Keep track of tasks that need cloud execution
        // We handle these separately because cloud scheduling needs
        // to account for wireless channel availability
        vector<Task*> cloud_entry_tasks;

        // First Pass: Handle Local Core Tasks
        // Process tasks that will run on local cores first
        for (auto* task : entry_tasks) {
            if (task->is_core_task) {
                // For local tasks, find the best core assignment
                // This considers both execution time and core availability
                auto choice = identify_optimal_local_core(*task);
                
                // Schedule the task on its chosen core
                // This updates both task state and core availability
                schedule_on_local_core(*task, choice.core, choice.start_time, choice.finish_time);
            } else {
                // Queue cloud tasks for later processing
                // We handle these together to manage wireless channel contention
                cloud_entry_tasks.push_back(task);
            }
        }

        // Second Pass: Handle Cloud Tasks
        // Process all cloud-bound tasks in sequence
        // This ensures proper management of the wireless channels
        for (auto* task : cloud_entry_tasks) {
            // Set the wireless sending ready time
            // This accounts for when the sending channel becomes available
            task->RT_ws = ws_ready;
            
            // Calculate timing for all three cloud execution phases
            // This includes sending, cloud computation, and receiving results
            auto timing = calculate_cloud_phases_timing(*task);
            
            // Record the complete schedule for this cloud task
            // This updates both task state and wireless channel availability
            schedule_on_cloud(*task, timing.send_ready, timing.send_finish, timing.cloud_ready, timing.cloud_finish, timing.receive_ready, timing.receive_finish);
        }
}

// Calculates ready times for non-entry tasks based on their predecessors
// Implements the ready time calculations from equations 3 and 4
void InitialTaskScheduler::calculate_non_entry_task_ready_times(Task &task) {
        // Calculate RT_l (ready time for local execution)
        // This implements Equation 3 from the paper
        int max_pred_finish_l_wr = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            // For each predecessor, consider both scenarios:
            // - FT_l: when it finishes if executed locally
            // - FT_wr: when its results arrive if executed in cloud
            int val = max(pred.FT_l, pred.FT_wr);
            if (val > max_pred_finish_l_wr) {
                max_pred_finish_l_wr = val;
            }
        }
        // Task can't start locally until all predecessors complete
        task.RT_l = max(max_pred_finish_l_wr, 0);

        // Calculate RT_ws (ready time for cloud sending)
        // This implements Equation 4 from the paper
        int max_pred_finish_l_ws = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            // For each predecessor, consider both scenarios:
            // - FT_l: when it finishes if executed locally
            // - FT_ws: when it finishes uploading if executed in cloud
            int val = max(pred.FT_l, pred.FT_ws);
            if (val > max_pred_finish_l_ws) {
                max_pred_finish_l_ws = val;
            }
        }
        // Task can't start uploading until:
        // - All predecessors complete
        // - Wireless sending channel becomes available
        task.RT_ws = max(max_pred_finish_l_ws, ws_ready);
}

// Schedules non-entry tasks with dependency and resource constraints
void InitialTaskScheduler::schedule_non_entry_tasks(vector<Task*> &non_entry_tasks) {
        // Process tasks in priority order
        for (auto* task : non_entry_tasks) {
            // First, calculate when this task could potentially start
            // based on its dependencies (Equations 3 and 4)
            calculate_non_entry_task_ready_times(*task);

            // Handle tasks initially marked for cloud execution
            if (!task->is_core_task) {
                // Calculate timing for cloud execution phases
                auto timing = calculate_cloud_phases_timing(*task);
                // Schedule task in cloud immediately
                schedule_on_cloud(*task, timing.send_ready, timing.send_finish,
                                timing.cloud_ready, timing.cloud_finish,
                                timing.receive_ready, timing.receive_finish);
            } 
            // Handle tasks initially marked for local execution
            else {
                // Find best local core option considering ready time
                auto local_choice = identify_optimal_local_core(*task, task->RT_l);
                
                // Calculate cloud timing for comparison
                auto timing = calculate_cloud_phases_timing(*task);
                int cloud_finish_time = timing.receive_finish;
                // Compare local and cloud execution times
                // Choose the option that finishes earlier
                if (local_choice.finish_time <= cloud_finish_time) {
                    // Local execution is faster, schedule on chosen core
                    schedule_on_local_core(*task, local_choice.core, local_choice.start_time, local_choice.finish_time);
                } else {
                    // Cloud execution is faster, switch to cloud
                    task->is_core_task = false;
                    schedule_on_cloud(*task, timing.send_ready, timing.send_finish,
                                    timing.cloud_ready, timing.cloud_finish,
                                    timing.receive_ready, timing.receive_finish);
                }
            }
        }
}

// Returns the final scheduling sequences for all execution units
// Each sequence represents the order of tasks assigned to that resource
vector<vector<int>> InitialTaskScheduler::get_sequences() const {
    return sequences;
}

// Implements the complete execution unit selection phase
vector<vector<int>> execution_unit_selection(vector<Task>& tasks) {
    // Initialize scheduler with 3 cores
    InitialTaskScheduler scheduler(tasks, 3);
    // Step 1: Get tasks ordered by priority
    // Higher priority means the task is on a longer path to completion
    // This helps minimize overall completion time
    vector<int> priority_orderered_tasks = scheduler.get_priority_ordered_tasks();
    // Step 2: Separate tasks into entry tasks and non-entry tasks
    // Entry tasks can start immediately while non-entry tasks must wait
    // This separation helps handle dependencies correctly
    auto [entry_tasks, non_entry_tasks] = scheduler.classify_entry_tasks(priority_orderered_tasks);
    // Step 3: Schedule all entry tasks first
    // These tasks form the foundation of the schedule
    scheduler.schedule_entry_tasks(entry_tasks);
    // Step 4: Schedule remaining tasks in priority order
    // Each task is placed on the best execution unit while maintainig dependency constraints
    scheduler.schedule_non_entry_tasks(non_entry_tasks);
    // Return the final sequences of tasks for each execution unit
    return scheduler.get_sequences();
}

// Constructs a new sequence when migrating a task to a different execution unit
vector<vector<int>> construct_sequence(
    vector<Task>& tasks,         // All tasks in the system
    int task_id,                      // Task being migrated
    int execution_unit,               // New execution unit for the task
    vector<vector<int>> original_sequence  // Current scheduling
) {
    // Get reference to the task being migrated
    Task &target_task = tasks[task_id - 1];
    // Determine when the task could start on its new execution unit
    // This depends on whether it's going to a core or to the cloud
    int target_task_rt = target_task.is_core_task ? 
                        target_task.RT_l :    // Ready time for local execution
                        target_task.RT_ws;    // Ready time for cloud sending

    // Remove the task from its current execution sequence
    int original_assignment = target_task.assignment;
    auto &old_seq = original_sequence[original_assignment];
    old_seq.erase(remove(old_seq.begin(), old_seq.end(), target_task.id), old_seq.end());
    // Get reference to the sequence where the task will be inserted
    auto &new_seq = original_sequence[execution_unit];
    // Collect start times of tasks already in the target sequence
    // This helps find the right insertion point
    vector<int> start_times;
    start_times.reserve(new_seq.size());
    for (int tid : new_seq) {
        Task &t = tasks[tid - 1];
        start_times.push_back(t.execution_unit_task_start_times[execution_unit]);
    }
    // Find where to insert the task based on its ready time
    auto it = lower_bound(start_times.begin(), start_times.end(), target_task_rt);
    int insertion_index = static_cast<int>(distance(start_times.begin(), it));
    // Insert the task at the appropriate position
    new_seq.insert(new_seq.begin() + insertion_index, target_task.id);
    // Update task's assignment information
    target_task.assignment = execution_unit;
    target_task.is_core_task = (execution_unit != 3);
    return original_sequence;
}

// Constructor initializes the scheduler with tasks and their current sequences
KernelScheduler::KernelScheduler(vector<Task>& tasks, vector<vector<int>>& sequences)
    : tasks(tasks), sequences(sequences)
{
    // Initialize ready times for local cores (RT_l)
    // These track when each core will next be available
    RT_ls = {0,0,0};

    // Initialize ready times for cloud execution phases
    // [0] = sending, [1] = computation, [2] = receiving
    cloud_phases_ready_times = {0,0,0};

    // Initialize task state tracking vectors
    tie(dependency_ready, sequence_ready) = initialize_task_state();
}

// Sets up initial task state tracking for dependencies and sequences
pair<vector<int>, vector<int>> KernelScheduler::initialize_task_state() {
        // Track how many predecessors each task is still waiting for
        vector<int> dependency_ready(tasks.size(), 0);
        for (size_t i = 0; i < tasks.size(); i++) {
            dependency_ready[i] = static_cast<int>(tasks[i].pred_tasks.size());
        }

        // Track which tasks are ready to execute in their sequences
        // -1 means not ready, 0 means ready to execute
        vector<int> sequence_ready(tasks.size(), -1);
        for (auto &seq : sequences) {
            if (!seq.empty()) {
                // First task in each sequence is initially ready
                sequence_ready[seq[0] - 1] = 0;
            }
        }
        return {dependency_ready, sequence_ready};
}

// Updates the readiness state of a task based on its dependencies and sequence position
void KernelScheduler::update_task_state(Task &task) {
        // Only update state for tasks not yet scheduled by kernel algorithm
        if (task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
            // First, handle dependency tracking
            // Count how many unscheduled predecessor tasks remain
            int unsched_preds = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                if (pred.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                    unsched_preds++;
                }
            }
            // Update the dependency counter for this task
            dependency_ready[task.id - 1] = unsched_preds;

            // Handle sequence-based readiness
            // Find this task in its current execution sequence
            for (auto &seq : sequences) {
                auto it = find(seq.begin(), seq.end(), task.id);
                if (it != seq.end()) {
                    // Calculate task's position in the sequence
                    int idx = static_cast<int>(distance(seq.begin(), it));
                    
                    if (idx > 0) {
                        // If task isn't first in sequence, check previous task
                        int prev_task_id = seq[idx - 1];
                        Task &prev_task = tasks[prev_task_id - 1];
                        // Task is ready (0) if previous task is scheduled
                        // Otherwise, it's waiting (1)
                        sequence_ready[task.id - 1] = (prev_task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) ? 1 : 0;
                    } else {
                        // If task is first in sequence, it's ready to go
                        sequence_ready[task.id - 1] = 0;
                    }
                    break;  // Found the sequence containing this task
                }
            }
        }
}

// Schedules a task for execution on a local core
void KernelScheduler::schedule_local_task(Task &task) {
        // Calculate when the task could theoretically start by checking
        // all of its prerequisites (predecessor tasks)
        if (task.pred_tasks.empty()) {
            // If there are no prerequisites, the task could start immediately
            task.RT_l = 0;
        } else {
            // Otherwise, need to wait for all prerequisites to complete
            // We track the latest completion time among all prerequisites
            int max_finish = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                // For each prerequisite, consider both possible execution paths:
                // - FT_l: when it finishes if it ran locally
                // - FT_wr: when its results arrive if it ran in the cloud
                int val = max(pred.FT_l, pred.FT_wr);
                if (val > max_finish) {
                    max_finish = val;
                }
            }
            // The task can't start before all prerequisites are ready
            task.RT_l = max_finish;
        }

        int core_index = task.assignment;
        // Clear any previous timing information
        task.execution_unit_task_start_times.assign(4, -1);
        // Calculate actual start time by considering both:
        // - When the task's prerequisites finish (RT_l)
        // - When the assigned core becomes available (RT_ls[core_index])
        int start_time = max(RT_ls[core_index], task.RT_l);
        // Record when this task will start on its assigned core
        task.execution_unit_task_start_times[core_index] = start_time;
        // Calculate when the task will finish
        // This adds the task's execution time on this specific core
        task.FT_l = start_time + task.core_execution_times[core_index];
        // Update when this core will next be available
        RT_ls[core_index] = task.FT_l;
        // Since this task runs locally, clear any cloud-related timing
        task.FT_ws = -1;  // No cloud sending phase
        task.FT_c = -1;   // No cloud computation phase
        task.FT_wr = -1;  // No cloud receiving phase
}

void KernelScheduler::schedule_cloud_task(Task &task) {
        // Phase 1: Calculate when can start sending data to the cloud
        if (task.pred_tasks.empty()) {
            task.RT_ws = 0;  // No prerequisites, can start immediately
        } else {
            // Find the latest finishing time among prerequisites
            int max_finish = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                // Consider both local execution (FT_l) and cloud sending (FT_ws)
                int val = max(pred.FT_l, pred.FT_ws);
                if (val > max_finish) {
                    max_finish = val;
                }
            }
            task.RT_ws = max_finish;
        }
        // Clear previous timing information
        task.execution_unit_task_start_times.assign(4, -1);
        // Phase 1 (continued): Schedule the sending phase
        int send_start = max(cloud_phases_ready_times[0], task.RT_ws);
        task.execution_unit_task_start_times[3] = send_start;
        task.FT_ws = send_start + task.cloud_execution_times[0];
        cloud_phases_ready_times[0] = task.FT_ws;
        // Phase 2: Schedule cloud computation
        int max_pred_c = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            // Find the latest cloud completion among prerequisites
            if (pred.FT_c > max_pred_c) {
                max_pred_c = pred.FT_c;
            }
        }
        // Cloud can't start until data arrives and prerequisites finish
        task.RT_c = max(task.FT_ws, max_pred_c);
        task.FT_c = max(cloud_phases_ready_times[1], task.RT_c) + task.cloud_execution_times[1];
        cloud_phases_ready_times[1] = task.FT_c;
        // Phase 3: Schedule result receiving
        task.RT_wr = task.FT_c;  // Can't receive until computation is done
        task.FT_wr = max(cloud_phases_ready_times[2], task.RT_wr) + 
                    task.cloud_execution_times[2];
        cloud_phases_ready_times[2] = task.FT_wr;
        // Clear local execution time since this task runs in the cloud
        task.FT_l = -1;
}

// Initializes the scheduling queue by identifying tasks that are ready to execute
deque<Task*> KernelScheduler::initialize_queue() {
        deque<Task*> dq;  // Will hold tasks ready for scheduling

        // Examine each task in the system
        for (auto &t : tasks) {
            // First check: Is this task ready in its sequence
            // sequence_ready[t.id - 1] == 0 means it's at the front of its sequence
            if (sequence_ready[t.id - 1] == 0) {
                // Second check: Are all prerequisites completed
                bool all_preds_sched = true;
                for (int pred_id : t.pred_tasks) {
                    Task &pred = tasks[pred_id - 1];
                    // A prerequisite must be fully scheduled
                    if (pred.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                        all_preds_sched = false;
                        break;
                    }
                }
                // If both checks pass, this task is ready to be scheduled
                if (all_preds_sched) {
                    dq.push_back(&t);
                }
            }
        }
        return dq;
}

// Implements the linear-time kernel algorithm
// This function reschedules tasks after migrations while maintaining dependencies
vector<Task>& kernel_algorithm(
    vector<Task>& tasks,
    vector<vector<int>>& sequences
) {
    // Initialize the scheduler with current tasks and sequences
    KernelScheduler scheduler(tasks, sequences);
    // Create initial queue of ready tasks
    // A task is ready when it's first in its sequence and prerequisites are done
    deque<Task*> queue = scheduler.initialize_queue();

    // Process tasks until queue is empty
    while (!queue.empty()) {
        // Get next ready task from front of queue
        Task* current_task = queue.front();
        queue.pop_front();
        // Mark task as scheduled by kernel algorithm
        current_task->is_scheduled = SchedulingState::KERNEL_SCHEDULED;
        // Schedule task according to its assignment (local or cloud)
        if (current_task->is_core_task) {
            scheduler.schedule_local_task(*current_task);
        } else {
            scheduler.schedule_cloud_task(*current_task);
        }
        // Update state of all tasks after this scheduling
        // This recalculates dependencies and sequence readiness
        for (auto &task : tasks) {
            scheduler.update_task_state(task);
        }
        // Check for newly ready tasks to add to queue
        for (auto &task : tasks) {
            int idx = task.id - 1;
            // A task is ready if:
            // 1. All dependencies are satisfied (dependency_ready[idx] == 0)
            // 2. It's first in its sequence (sequence_ready[idx] == 0)
            // 3. It hasn't been scheduled yet
            if (scheduler.dependency_ready[idx] == 0 &&
                scheduler.sequence_ready[idx] == 0 &&
                task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                // Check if task is already in queue to avoid duplicates
                bool in_queue = false;
                for (auto* tptr : queue) {
                    if (tptr == &task) {
                        in_queue = true;
                        break;
                    }
                }
                // Add newly ready task to queue if not already present
                if (!in_queue) {
                    queue.push_back(&task);
                }
            }
        }
    }
    // Reset scheduling state for next iteration
    for (auto &task : tasks) {
        task.is_scheduled = SchedulingState::UNSCHEDULED;
    }
    return tasks;
}

// Defines how to compare two migration scenarios
bool MigrationKey::operator<(const MigrationKey& other) const {
    // Compare task indices first
    if (task_idx != other.task_idx)
        return task_idx < other.task_idx;
    // If tasks are the same, compare target execution units
    if (target_execution_unit != other.target_execution_unit)
        return target_execution_unit < other.target_execution_unit;
    // Finally, compare the entire assignment state
    return assignments < other.assignments;
}

// Creates a cache key for a specific migration scenario
// This captures the complete state needed to identify unique migrations
MigrationKey generate_cache_key(
    const vector<Task>& tasks,     // All tasks in the system
    int task_idx,                       // Task being migrated
    int target_execution_unit           // Destination for the task
) {
    MigrationKey key;
    key.task_idx = task_idx;
    key.target_execution_unit = target_execution_unit;
    // Record current assignments of all tasks
    key.assignments.reserve(tasks.size());
    for (const auto& t : tasks) {
        key.assignments.push_back(t.assignment);
    }
    return key;
}

// Evaluates the impact of moving a task to a new execution unit
// Returns both completion time and energy consumption for the migration
pair<int,double> evaluate_migration(
    vector<Task>& tasks,                   // All tasks in system
    const vector<vector<int>>& seqs,  // Current execution sequences
    int task_idx,                               // Task considering moving
    int target_execution_unit,                  // Target Location
    map<MigrationKey, pair<int,double>>& migration_cache,  // Cache of previous evaluations
    const vector<int>& core_powers,      // Power consumption of cores
    double cloud_sending_power                    // Power for cloud communication
) {
    //Check if already evaluated this exact migration scenario
    MigrationKey cache_key = generate_cache_key(tasks, task_idx, target_execution_unit);
    auto it = migration_cache.find(cache_key);
    if (it != migration_cache.end()) {
        // If seen this before, return cached result
        return it->second;
    }

    // Create copies of the current state to simulate the migration
    vector<vector<int>> sequence_copy = seqs;
    vector<Task> tasks_copy = tasks;

    // Simulate the task migration using the linear-time rescheduling algorithm
    sequence_copy = construct_sequence(tasks_copy, task_idx + 1, target_execution_unit, sequence_copy);
    kernel_algorithm(tasks_copy, sequence_copy);

    // Calculate both completion time and energy consumption for the new schedule after migration
    int migration_T = total_time(tasks_copy);
    double migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power);

    // Cache the results for future reference
    migration_cache[cache_key] = make_pair(migration_T, migration_E);
    return {migration_T, migration_E};
}

// Initializes a matrix of valid migration choices for each task
vector<array<bool,4>> initialize_migration_choices(
    const vector<Task>& tasks
) {
    // Create a matrix where each task has 4 possible destinations
    // (3 cores + 1 cloud option), initialized to all false
    vector<array<bool,4>> migration_choices(
        tasks.size(), 
        array<bool,4>{false,false,false,false}
    );

    // For each task
    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        // If task is currently in the cloud (assignment == 3)
        if (task.assignment == 3) {
            // Consider migrations to any core or staying in cloud
            for (int j = 0; j < 4; j++) {
                migration_choices[i][j] = true;
            }
        } else {
            // For tasks on local cores, only consider their current assignment as valid initially
            // This creates a baseline before exploring other options
            migration_choices[i][task.assignment] = true;
        }
    }
    return migration_choices;
}

// Sort by efficiency (higher efficiency = higher priority)
bool MigrationCandidate::operator<(const MigrationCandidate& other) const {
    return neg_efficiency > other.neg_efficiency;
}

TaskMigrationState* identify_optimal_migration(
    const vector<tuple<int,int,int,double>>& migration_trials_results,
    int T_final,    // Current completion time
    double E_total, // Current energy consumption
    int T_max       // Maximum allowed completion time
) {
    // Phase 1: Look for migrations that reduce energy without increasing time
    double best_energy_reduction = 0.0;
    TaskMigrationState* best_migration_state = nullptr;

    // Evaluate all potential migrations
    for (auto &res : migration_trials_results) {
        int task_idx, resource_idx, time_int;
        double energy;
        tie(task_idx, resource_idx, time_int, energy) = res;
        int time = time_int;

        // Skip migrations that violate maximum time constraint
        if (time > T_max) {
            continue;
        }

        // Calculate potential energy savings
        double energy_reduction = E_total - energy;
        // If can save energy without increasing time, this is ideal
        if (time <= T_final && energy_reduction > 0.0) {
            if (energy_reduction > best_energy_reduction) {
                best_energy_reduction = energy_reduction;
                if (best_migration_state) {
                    delete best_migration_state;
                }
                best_migration_state = new TaskMigrationState{
                    time, energy, best_energy_reduction,
                    task_idx + 1, resource_idx + 1
                };
            }
        }
    }
    // If found a migration that reduces energy without increasing time, use it
    if (best_migration_state) {
        return best_migration_state;
    }
    // Phase 2: Consider migrations that trade time for energy savings
    priority_queue<MigrationCandidate> migration_candidates;
    // Evaluate all migrations again, this time considering time-energy trade-offs
    for (auto &res : migration_trials_results) {
        int task_idx, resource_idx, time_int;
        double energy;
        tie(task_idx, resource_idx, time_int, energy) = res;
        int time = time_int;

        if (time > T_max) {
            continue;
        }
        double energy_reduction = E_total - energy;
        if (energy_reduction > 0.0) {
            // Calculate efficiency as energy savings per unit time increase
            int time_increase = max(0, time - T_final);
            double efficiency;
            if (time_increase == 0) {
                efficiency = numeric_limits<double>::infinity();
            } else {
                efficiency = energy_reduction / static_cast<double>(time_increase);
            }
            migration_candidates.push(MigrationCandidate{-efficiency, task_idx, resource_idx, time, energy});
        }
    }

    // If no beneficial migrations found, return null
    if (migration_candidates.empty()) {
        return nullptr;
    }
    // Return the most efficient migration
    MigrationCandidate best = migration_candidates.top();
    double efficiency = -best.neg_efficiency;
    return new TaskMigrationState{
        best.time, best.energy, efficiency,
        best.task_idx + 1, best.resource_idx + 1
    };
}

// Implements the task scheduling optimization
// Iteratively improves energy consumption through strategic task migrations
pair<vector<Task>, vector<vector<int>>>
optimize_task_scheduling(
    vector<Task> tasks,              // Tasks to be scheduled
    vector<vector<int>> sequence, // Initial task sequences
    int T_final,                          // Target completion time
    vector<int> core_powers,  // Power consumption of cores
    double cloud_sending_power           // Power for cloud communication
) {
    // Cache to avoid re-evaluating identical migration scenarios
    map<MigrationKey, pair<int,double>> migration_cache;
    // Calculate initial energy consumption as baseline
    double current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power);
    // Continue optimizing as long as can find improvements
    bool energy_improved = true;
    while (energy_improved) {
        double previous_iteration_energy = current_iteration_energy;
        int current_time = total_time(tasks);
        // Allow completion time to increase up to 50% above target
        int T_max = static_cast<int>(floor(T_final * 1.5));
        // Initialize matrix of potential migrations to evaluate
        auto migration_choices = initialize_migration_choices(tasks);
        // Collect results from evaluating all possible migrations
        vector<tuple<int,int,int,double>> migration_trials_results;
        for (size_t task_idx = 0; task_idx < tasks.size(); task_idx++) {
            for (int possible_execution_unit = 0; possible_execution_unit < 4; 
                 possible_execution_unit++) {
                // Skip migrations marked as invalid
                if (migration_choices[task_idx][possible_execution_unit]) {
                    continue;
                }
                // Evaluate impact of this potential migration
                auto [migration_trial_time, migration_trial_energy] = evaluate_migration(tasks, sequence, static_cast<int>(task_idx),possible_execution_unit, migration_cache,core_powers, cloud_sending_power);
                migration_trials_results.push_back(make_tuple(
                    static_cast<int>(task_idx), possible_execution_unit,
                    migration_trial_time, migration_trial_energy));
            }
        }

        // Find the best migration among all candidates
        TaskMigrationState* best_migration = identify_optimal_migration(
            migration_trials_results,
            current_time,
            previous_iteration_energy,
            T_max
        );

        // If no beneficial migration found, done optimizing
        if (!best_migration) {
            energy_improved = false;
            break;
        }

        // Apply the best migration and update schedule
        sequence = construct_sequence(
            tasks,
            best_migration->task_index,
            best_migration->target_execution_unit - 1,
            sequence
        );

        // Reschedule all tasks after the migration
        kernel_algorithm(tasks, sequence);

        // Check if actually improved energy consumption
        current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power);
        energy_improved = (current_iteration_energy < previous_iteration_energy);

        delete best_migration;

        // Prevent cache from growing too large
        if (migration_cache.size() > 1000) {
            migration_cache.clear();
        }
    }

    return {tasks, sequence};
}

void print_schedule_tasks(const vector<Task>& tasks) {
    static const char* ASSIGNMENT_MAPPING[] = {
        "Core 1", "Core 2", "Core 3", "Cloud"
    };

    const int width_id = 8;
    const int width_assignment = 12;
    const int width_start = 12;
    const int width_execwindow = 25;
    const int width_send = 20;
    const int width_cloud = 20;
    const int width_receive = 20;

    cout << "\nTask Schedule:\n";
    cout << string(130, '-') << "\n";

    cout << left
              << setw(width_id) << "TaskID"
              << setw(width_assignment) << "Assignment"
              << setw(width_start) << "StartTime"
              << setw(width_execwindow) << "ExecWindow(Core)"
              << setw(width_send) << "SendPhase(Cloud)"
              << setw(width_cloud) << "CloudPhase(Cloud)"
              << setw(width_receive) << "ReceivePhase(Cloud)"
              << "\n";

    cout << string(130, '-') << "\n";

    for (const auto& task : tasks) {
        string assignment_str;
        if (task.assignment >= 0 && task.assignment <= 3) {
            assignment_str = ASSIGNMENT_MAPPING[task.assignment];
        } else if (task.assignment == -2) {
            assignment_str = "Not Scheduled";
        } else {
            assignment_str = "Unknown";
        }

        string execution_window = "-";
        string send_phase = "-";
        string cloud_phase = "-";
        string receive_phase = "-";

        int display_start_time = -1;

        if (task.is_core_task && task.assignment >= 0 &&
            static_cast<size_t>(task.assignment) < task.execution_unit_task_start_times.size()) {
            int start_time = task.execution_unit_task_start_times[task.assignment];
            int end_time = start_time + task.core_execution_times[task.assignment];
            execution_window = to_string(start_time) + "=>" + to_string(end_time);
            display_start_time = start_time;
        }

        if (!task.is_core_task && task.execution_unit_task_start_times.size() > 3) {
            int send_start = task.execution_unit_task_start_times[3];
            int send_end = send_start + task.cloud_execution_times[0];
            send_phase = to_string(send_start) + "=>" + to_string(send_end);

            int RT_c_val = task.RT_c;
            int cloud_end = RT_c_val + task.cloud_execution_times[1];
            cloud_phase = to_string(RT_c_val) + "=>" + to_string(cloud_end);

            int RT_wr_val = task.RT_wr;
            int receive_end = RT_wr_val + task.cloud_execution_times[2];
            receive_phase = to_string(RT_wr_val) + "=>" + to_string(receive_end);

            display_start_time = send_start;
        }

        cout << left
                  << setw(width_id) << task.id
                  << setw(width_assignment) << assignment_str
                  << setw(width_start) << ((display_start_time >= 0) ? to_string(display_start_time) : "-")
                  << setw(width_execwindow) << execution_window
                  << setw(width_send) << send_phase
                  << setw(width_cloud) << cloud_phase
                  << setw(width_receive) << receive_phase
                  << "\n";
    }

    cout << string(130, '-') << "\n";
}

tuple<bool, vector<string>> validate_schedule_constraints(
    const vector<Task>& tasks
) {
    // Track any constraint violations found
    vector<string> violations;
    map<int, size_t> id_to_index;
    for (size_t i = 0; i < tasks.size(); ++i) {
        id_to_index[tasks[i].id] = i;
    }
    auto get_task = [&](int task_id) -> const Task& {
        return tasks[id_to_index.at(task_id)];
    };

    // Organize tasks by execution location for efficient validation
    vector<const Task*> cloud_tasks;
    vector<const Task*> core_tasks;
    for (auto &t : tasks) {
        if (t.is_core_task) {
            core_tasks.push_back(&t);
        } else {
            cloud_tasks.push_back(&t);
        }
    }

    // Validation 1: Wireless Sending Channel
    // Ensures no overlap in cloud data transmission
    {
        auto sorted = cloud_tasks;
        sort(sorted.begin(), sorted.end(), 
                 [](const Task* a, const Task* b) {
            return a->execution_unit_task_start_times[3] < 
                   b->execution_unit_task_start_times[3];
        });
        // Check for overlapping transmissions
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_ws > next_task->execution_unit_task_start_times[3]) {
                violations.push_back("Wireless Sending Channel Conflict...");
            }
        }
    }

    // Validation 2: Cloud Computing
    // Verifies proper sequencing of cloud computations
    {
        auto sorted = cloud_tasks;
        sort(sorted.begin(), sorted.end(), 
                 [](const Task* a, const Task* b) {
            return a->RT_c < b->RT_c;
        });
        // Check for computation overlap
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_c > next_task->RT_c) {
                violations.push_back("Cloud Computing Conflict...");
            }
        }
    }

    // Validation 3: Wireless Receiving Channel
    // Ensures no overlap in result reception
    {
        auto sorted = cloud_tasks;
        sort(sorted.begin(), sorted.end(), 
                 [](const Task* a, const Task* b) {
            return a->RT_wr < b->RT_wr;
        });
        // Check for receiving conflicts
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_wr > next_task->RT_wr) {
                violations.push_back("Wireless Receiving Channel Conflict...");
            }
        }
    }

    // Validation 4: Task Dependencies
    // Verifies all predecessor-successor relationships
    {
        for (auto &task : tasks) {
            if (!task.is_core_task) {
                // Handle cloud task dependencies
                int task_send_start = task.execution_unit_task_start_times[3];
                for (int pred_id : task.pred_tasks) {
                    const Task &pred = get_task(pred_id);
                    if (pred.is_core_task) {
                        // Core to cloud dependencies
                        if (pred.FT_l > task_send_start) {
                            violations.push_back("Core-Cloud Dependency Violation...");
                        }
                    } else {
                        // Cloud to cloud dependencies
                        if (pred.FT_ws > task_send_start) {
                            violations.push_back("Cloud Pipeline Dependency Violation...");
                        }
                    }
                }
            } else {
                // Handle core task dependencies
                int core_id = task.assignment;
                int task_start = task.execution_unit_task_start_times[core_id];
                for (int pred_id : task.pred_tasks) {
                    const Task &pred = get_task(pred_id);
                    int pred_finish = pred.is_core_task ? pred.FT_l : pred.FT_wr;
                    if (pred_finish > task_start) {
                        violations.push_back("Core Task Dependency Violation...");
                    }
                }
            }
        }
    }

    // Validation 5: Core Execution
    // Checks for conflicts on each core
    {
        for (int core_id = 0; core_id < 3; ++core_id) {
            vector<const Task*> core_specific;
            for (auto t : core_tasks) {
                if (t->assignment == core_id) {
                    core_specific.push_back(t);
                }
            }
            sort(core_specific.begin(), core_specific.end(),
                     [=](const Task* a, const Task* b) {
                return a->execution_unit_task_start_times[core_id] < 
                       b->execution_unit_task_start_times[core_id];
            });
            // Check for execution overlaps
            for (size_t i = 0; i + 1 < core_specific.size(); ++i) {
                const Task* current = core_specific[i];
                const Task* next_task = core_specific[i + 1];
                if (current->FT_l > 
                    next_task->execution_unit_task_start_times[core_id]) {
                    violations.push_back("Core Execution Conflict...");
                }
            }
        }
    }

    bool is_valid = violations.empty();
    return {is_valid, violations};
}

void print_schedule_validation_report(const vector<Task>& tasks) {
    auto [is_valid, violations] = validate_schedule_constraints(tasks);

    cout << "\nSchedule Validation Report\n";
    cout << string(50, '=') << "\n";

    if (is_valid) {
        cout << "Schedule is valid with all constraints satisfied!\n";
    } else {
        cout << "Found constraint violations:\n";
        for (const auto& violation : violations) {
            cout << violation << "\n";
        }
    }
}

// Creates a readable display of how tasks are scheduled across execution units (local cores and cloud)
void print_schedule_sequences(const vector<vector<int>>& sequences) {
    cout << "\nExecution Sequences:\n";
    cout << string(40, '-') << "\n";
    // Iterate through each execution unit (3 cores + cloud)
    for (size_t i = 0; i < sequences.size(); i++) {
        // Label each sequence appropriately - either core number or cloud
        string label = (i < 3) ? "Core " + to_string(i + 1) : "Cloud";
        cout << setw(12) << left << label << ": ";
        // Print task sequence in array format
        cout << "[";
        for (size_t j = 0; j < sequences[i].size(); j++) {
            if (j > 0) cout << ", ";
            cout << sequences[i][j];
        }
        cout << "]\n";
    }
}

// Creates a complete task graph with execution times and dependencies
vector<Task> create_task_graph(
    const vector<int>& task_ids,                    // Unique IDs for each task
    const map<int, array<int,3>>& core_exec_times,  // Execution times on local cores
    const array<int,3>& cloud_exec_times,           // Cloud execution phase times
    const vector<pair<int,int>>& edges         // Task dependencies
) {
    // Initialize all tasks with their execution time requirements
    vector<Task> tasks;
    tasks.reserve(task_ids.size());  // Optimize memory allocation
    
    // Create each task with its timing information
    for (int tid : task_ids) {
        tasks.emplace_back(tid, core_exec_times, cloud_exec_times);
    }
    // Create a mapping from task IDs to their positions in the vector
    // This makes it efficient to establish task relationships
    map<int,int> id_to_index;
    for (size_t i = 0; i < tasks.size(); ++i) {
        id_to_index[tasks[i].id] = static_cast<int>(i);
    }
    // Establish predecessor and successor relationships
    // This creates the directed edges in the task graph
    for (auto &edge : edges) {
        int from = edge.first;   // Source task ID
        int to   = edge.second;  // Destination task ID
        // Record that source task must complete before destination task starts
        tasks[id_to_index[from]].succ_tasks.push_back(to);
        tasks[id_to_index[to]].pred_tasks.push_back(from);
    }

    return tasks;
}

int main() {
    static const map<int, array<int,3>> core_execution_times = {
        {1, {9, 7, 5}}, {2, {8, 6, 5}}, {3, {6, 5, 4}}, {4, {7, 5, 3}},
        {5, {5, 4, 2}}, {6, {7, 6, 4}}, {7, {8, 5, 3}}, {8, {6, 4, 2}},
        {9, {5, 3, 2}}, {10,{7,4,2}},  {11,{10,7,4}}, {12,{11,8,5}},
        {13,{9,6,3}}, {14,{12,8,4}}, {15,{10,7,3}}, {16,{11,7,4}},
        {17,{9,6,3}}, {18,{12,8,5}}, {19,{10,7,4}}, {20,{11,8,5}}
    };

    static const array<int,3> cloud_execution_times = {3, 1, 1};

    vector<int> ten_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<int> twenty_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    // Graph 1
    vector<pair<int,int>> graph1_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,8}, {2,9},
        {3,7},
        {4,8}, {4,9},
        {5,9},
        {6,8},
        {7,10},
        {8,10},
        {9,10}
    };

    //Graph 2
    vector<pair<int,int>> graph2_edges = {
        {1,2}, {1,3},
        {2,4}, {2,5},
        {3,5}, {3,6},
        {4,6},
        {5,7},
        {6,7}, {6,8},
        {7,8}, {7,9},
        {8,10},
        {9,10}
    };

    //Graph 3
    vector<pair<int,int>> graph3_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,7}, {2,8},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 4
    vector<pair<int,int>> graph4_edges = {
        {1,7},
        {2,7},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 5
    vector<pair<int,int>> graph5_edges = {
        {1,4}, {1,5}, {1,6},
        {2,7}, {2,8},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18},
        {15,19},
        {16,19},
        {18,20},
    };

    vector<vector<Task>> all_graphs;
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph1_edges));
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph2_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph3_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph4_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph5_edges));

    // Process each graph
    for (size_t i = 0; i < all_graphs.size(); ++i) {
        cout << "\nProcessing Graph " << (i+1) << ":\n";
        auto &tasks = all_graphs[i];

        // Step 1: Initial Scheduling Phase
        primary_assignment(tasks, 3);
        task_prioritizing(tasks);
        auto sequence = execution_unit_selection(tasks);

        // Step 2: Evaluate Initial Scheduling Results
        int T_final = total_time(tasks);
        double E_total = total_energy(tasks, {1,2,4}, 0.5);
        cout << "\nINITIAL SCHEDULING RESULTS:\n";
        cout << "----------------\n";
        cout << "INITIAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final << "\n";
        cout << "INITIAL APPLICATION ENERGY CONSUMPTION: " << E_total << "\n";
        cout << "INITIAL TASK SCHEDULE:\n";
        print_schedule_tasks(tasks);
        print_schedule_validation_report(tasks);
        print_schedule_sequences(sequence);
        cout << "\n\n----------------------------------------\n\n";

        // Step 3: Energy Optimization Phase/ Kernel Scheduling
        auto [tasks2, sequence2] = optimize_task_scheduling(tasks, sequence, T_final, {1,2,4}, 0.5);

        // Step 4: Evaluate Final Scheduling Results
        int T_final_after = total_time(tasks2);
        double E_final = total_energy(tasks2, {1,2,4}, 0.5);
        cout << "FINAL SCHEDULING RESULTS:\n";
        cout << "--------------\n";
        cout << "MAXIMUM APPLICATION COMPLETION TIME: " << T_final*1.5 << "\n";
        cout << "FINAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final_after << "\n";
        cout << "FINAL APPLICATION ENERGY CONSUMPTION: " << E_final << "\n";
        cout << "FINAL TASK SCHEDULE:\n";
        print_schedule_tasks(tasks2);
        print_schedule_validation_report(tasks2);
        print_schedule_sequences(sequence2);
    }

    return 0;
}
