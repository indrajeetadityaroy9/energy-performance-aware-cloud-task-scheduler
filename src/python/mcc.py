from copy import deepcopy
import bisect
from dataclasses import dataclass
from collections import deque
import numpy as np
from heapq import heappush, heappop
from enum import Enum

# Dictionary storing execution times for tasks on different cores
# Key: task ID (1-20)
# Value: List of execution times [core1_time, core2_time, core3_time]
# This implements T_i^l_k from Section II.B of the paper
core_execution_times = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2],
    11: [12, 3, 3],
    12: [12, 8, 4],
    13: [11, 3, 2],
    14: [12, 11, 4],
    15: [13, 4, 2],
    16: [9, 7, 3],
    17: [9, 3, 3],
    18: [13, 9, 2],
    19: [10, 5, 3],
    20: [12, 5, 4]
}

# Cloud execution parameters from Section II.B of the paper:
# - T_send: Time to send task to cloud (T_i^s in paper)
# - T_cloud: Cloud computation time (T_i^c in paper) 
# - T_receive: Time to receive results (T_i^r in paper)
cloud_execution_times = [3, 1, 1]


class SchedulingState(Enum):
    # States corresponding to the two-step algorithm in Section III:
    UNSCHEDULED = 0  # Task not yet processed
    SCHEDULED = 1  # After initial scheduling (Step 1: minimal-delay scheduling)
    KERNEL_SCHEDULED = 2  # After task migration (Step 2: energy optimization)


@dataclass
class TaskMigrationState:
    # Class to track task migration decisions as described in Section III.B
    time: float  # T_total: Total completion time after migration
    energy: float  # E_total: Total energy consumption after migration
    efficiency: float  # Energy reduction per unit time (used for migration decisions)
    task_index: int  # v_tar: Task selected for migration
    target_execution_unit: int  # k_tar: Target execution unit (core or cloud)


class Task(object):
    def __init__(self, id, pred_tasks=None, succ_task=None):
        # Basic task graph structure (Section II.A)
        self.id = id  # Task identifier vi in directed acyclic graph G=(V,E)
        self.pred_tasks = pred_tasks or []  # pred(vi): Set of immediate predecessors in task graph
        self.succ_task = succ_task or []  # succ(vi): Set of immediate successors in task graph

        # Task execution timing parameters (Section II.B)
        # Local execution times for each core k (equation 7)
        self.core_execution_times = core_execution_times[id]  # Ti,k^l: execution time on k-th core

        # Cloud execution phases (Section II.B):
        # [Ti^s: RF sending phase time,
        #  Ti^c: cloud computing phase time, 
        #  Ti^r: RF receiving phase time]
        self.cloud_execution_times = cloud_execution_times

        # Task completion timing parameters (Section II.C)
        # Finish Times for different execution units:
        self.FT_l = 0  # FTi^l:  Local core finish time
        self.FT_ws = 0  # FTi^ws: Wireless sending finish time
        self.FT_c = 0  # FTi^c:  Cloud computation finish time
        self.FT_wr = 0  # FTi^wr: Wireless receiving finish time

        # Ready Times (equations 3-6):
        self.RT_l = -1  # RTi^l:  Ready time for local execution
        self.RT_ws = -1  # RTi^ws: Ready time for wireless sending
        self.RT_c = -1  # RTi^c:  Ready time for cloud execution
        self.RT_wr = -1  # RTi^wr: Ready time for receiving results

        # Task scheduling parameters (Section III)
        # Priority score from equation (15) used in initial scheduling
        self.priority_score = None  # priority(vi) = wi + max(priority(vj))

        # Execution assignment (Section II.B):
        # ki = 0: offloaded to cloud
        # ki > 0: executed on local core ki
        self.assignment = -2

        # Flag to track if task is assigned to local core vs cloud
        self.is_core_task = False

        # Start times for possible execution units (used in scheduling)
        # Index 0: cloud, Indices 1-3: local cores
        self.execution_unit_task_start_times = [-1, -1, -1, -1]

        # Final completion time for the task
        self.execution_finish_time = -1

        # Current state in two-step scheduling algorithm (Section III):
        # - UNSCHEDULED: Initial state
        # - SCHEDULED: After initial minimal-delay scheduling
        # - KERNEL_SCHEDULED: After energy optimization
        self.is_scheduled = SchedulingState.UNSCHEDULED


def total_time(tasks):
    # Implementation of total completion time calculation T_total from equation (10):
    # T_total = max(max(FTi^l, FTi^wr))
    #           vi∈exit_tasks

    # Find the maximum completion time among all exit tasks
    return max(
        # For each exit task vi, compute max(FTi^l, FTi^wr) where:
        # - FTi^l: Finish time if task executes on local core
        # - FTi^wr: Finish time if task executes on cloud (when results are received)
        # If FTi^l = 0, task executed on cloud
        # If FTi^wr = 0, task executed locally
        max(task.FT_l, task.FT_wr)

        # Only consider exit tasks (tasks with no successors)
        # This implements the "exit tasks" condition from equation (10)
        for task in tasks
        if not task.succ_task
    )


def calculate_energy_consumption(task, core_powers, cloud_sending_power):
    # Calculate energy consumption for a single task vi
    # Based on equations (7) and (8) from Section II.D

    # For tasks executing on local cores (ki > 0)
    if task.is_core_task:
        # Implement equation (7): Ei,k^l = Pk × Ti,k^l where:
        # - Pk: Power consumption of k-th core (core_powers[k])
        # - Ti,k^l: Execution time of task vi on core k
        # - task.assignment: Selected core k for execution
        return core_powers[task.assignment] * task.core_execution_times[task.assignment]

    # For tasks offloaded to cloud (ki = 0)
    else:
        # Implement equation (8): Ei^c = P^s × Ti^s where:
        # - P^s: Power consumption of RF component for sending
        # - Ti^s: Time required to send task vi to cloud
        return cloud_sending_power * task.cloud_execution_times[0]


def total_energy(tasks, core_powers, cloud_sending_power):
    # Calculate total energy consumption E^total
    # Implements equation (9) from Section II.D:
    # E^total = ∑(i=1 to N) Ei where:
    # - N: Total number of tasks
    # - Ei: Energy consumption of task vi
    # - Ei equals either Ei,k^l (local) or Ei^c (cloud)
    return sum(
        calculate_energy_consumption(task, core_powers, cloud_sending_power)
        for task in tasks
    )


def primary_assignment(tasks):
    """
    Implements the "Primary Assignment" phase described in Section III.A.1.
    This is the first phase of the initial scheduling algorithm that determines
    which tasks should be considered for cloud execution.
    """
    for task in tasks:
        # Calculate T_i^l_min (minimum local execution time)
        # Implements equation (11): T_i^l_min = min(1≤k≤K) T_i,k^l
        # where K is the number of local cores
        # This represents the best-case local execution scenario
        # by choosing the fastest available core
        t_l_min = min(task.core_execution_times)

        # Calculate T_i^re (remote execution time)
        # Implements equation (12): T_i^re = T_i^s + T_i^c + T_i^r
        # This represents total time for cloud execution including:
        # T_i^s: Time to send task specification and input data
        # T_i^c: Time for cloud computation
        # T_i^r: Time to receive results
        t_re = (task.cloud_execution_times[0] +  # T_i^s (send)
                task.cloud_execution_times[1] +  # T_i^c (cloud)
                task.cloud_execution_times[2])  # T_i^r (receive)

        # Task assignment decision
        # If T_i^re < T_i^l_min, offloading to cloud will save time
        # This implements the cloud task selection criteria from Section III.A.1:
        if t_re < t_l_min:
            task.is_core_task = False  # Mark as cloud task
        else:
            task.is_core_task = True  # Mark for local execution


def task_prioritizing(tasks):
    """
    Implements the "Task Prioritizing" phase described in Section III.A.2.
    Calculates priority levels for each task to determine scheduling order.
    """
    w = [0] * len(tasks)
    # Step 1: Calculate computation costs (wi) for each task
    for i, task in enumerate(tasks):
        if not task.is_core_task:
            # For cloud tasks:
            # Implement equation (13): wi = Ti^re
            # where Ti^re is total remote execution time including:
            # - Data sending time (Ti^s)
            # - Cloud computation time (Ti^c)
            # - Result receiving time (Ti^r)
            w[i] = (task.cloud_execution_times[0] +  # Ti^s
                    task.cloud_execution_times[1] +  # Ti^c
                    task.cloud_execution_times[2])  # Ti^r
        else:
            # For local tasks:
            # Implement equation (14): wi = avg(1≤k≤K) Ti,k^l
            # Average computation time across all K cores
            w[i] = sum(task.core_execution_times) / len(task.core_execution_times)

    # Cache for memoization of priority calculations
    # This optimizes recursive calculations for tasks in cycles
    computed_priority_scores = {}

    def calculate_priority(task):
        """
        Recursive implementation of priority calculation.
        Implements equations (15) and (16) from the paper.
        """
        # Memoization check
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        # Base case: Exit tasks
        # Implement equation (16): priority(vi) = wi for vi ∈ exit_tasks
        if task.succ_task == []:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]

        # Recursive case: Non-exit tasks
        # Implement equation (15): 
        # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        # This represents length of critical path from task to exit
        max_successor_priority = max(calculate_priority(successor)
                                     for successor in task.succ_task)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks using recursive algorithm
    for task in tasks:
        calculate_priority(task)

    # Update priority scores in task objects
    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]


class InitialTaskScheduler:
    """
    Implements the initial scheduling algorithm described in Section III.A.
    This is Step One of the two-step algorithm, focusing on minimal-delay scheduling.
    """

    def __init__(self, tasks, num_cores=3):
        """
        Initialize scheduler with tasks and resources.
        Tracks timing for both local cores and cloud communication channels.
        """
        self.tasks = tasks
        self.k = num_cores  # K cores from paper

        # Resource timing tracking (Section II.B and II.C)
        self.core_earliest_ready = [0] * self.k  # When each core becomes available
        self.ws_ready = 0  # Next available time for RF sending channel
        self.wr_ready = 0  # Next available time for RF receiving channel

        # Sk sequence sets from Section III.B
        # Tracks task execution sequences for each resource (cores + cloud)
        self.sequences = [[] for _ in range(self.k + 1)]

    def get_priority_ordered_tasks(self):
        """
        Orders tasks by priority scores calculated in task_prioritizing().
        Implements ordering described in Section III.A.2, equation (15).
        """
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)  # Higher priority first
        return [item[1] for item in task_priority_list]

    def classify_entry_tasks(self, priority_order):
        """
        Separates tasks into entry and non-entry tasks while maintaining priority order.
        Implements task classification from Section II.A and III.A.3 of the paper:
        - Entry tasks: Starting points in the task graph with no dependencies
        - Non-entry tasks: Tasks that must wait for predecessors to complete
        """
        entry_tasks = []
        non_entry_tasks = []

        # Process tasks in priority order (from equation 15)
        # This ensures high-priority tasks are scheduled first
        for task_id in priority_order:
            task = self.tasks[task_id - 1]

            # Check if task has predecessors (pred(vi) from paper)
            if not task.pred_tasks:
                # Entry tasks have no predecessors and can start immediately
                # These correspond to v1 in Figure 1 of the paper
                entry_tasks.append(task)
            else:
                # Non-entry tasks must wait for predecessors to complete
                # Their ready times (RT) will be calculated based on predecessor finish times
                non_entry_tasks.append(task)

        return entry_tasks, non_entry_tasks

    def identify_optimal_local_core(self, task, ready_time=0):
        """
        Finds optimal local core assignment for a task to minimize finish time.
        Implements the local core selection logic from Section III.A.3 for tasks
        that will be executed locally rather than offloaded to the cloud.
        """
        # Initialize with worst-case values
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')

        # Try each available core k (1 ≤ k ≤ K)
        for core in range(self.k):
            # Calculate earliest possible start time on this core
            # Must be after both:
            # 1. Task's ready time RTi^l (based on predecessors)
            # 2. Core's earliest available time (when previous task finishes)
            start_time = max(ready_time, self.core_earliest_ready[core])

            # Calculate finish time FTi^l using:
            # - Start time determined above
            # - Task's execution time on this core (Ti,k^l)
            finish_time = start_time + task.core_execution_times[core]

            # Keep track of core that gives earliest finish time
            # This implements the "minimizes the task's finish time"
            # criteria from Section III.A.3
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core
                best_start_time = start_time

        return best_core, best_start_time, best_finish_time

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """
        Assigns a task to a local core and updates all relevant timing information.
        Implements local task scheduling from Section II.C.1 of the paper.
        """
        # Set task finish time on local core (FTi^l)
        # This is used in equation (10) for total completion time
        task.FT_l = finish_time
        # Set overall execution finish time
        # Used for precedence constraints and scheduling subsequent tasks
        task.execution_finish_time = finish_time
        # Initialize execution start times array
        # Index 0 to k: local cores
        # Index k+1: cloud
        # -1 indicates not scheduled on that unit
        task.execution_unit_task_start_times = [-1] * (self.k + 1)
        # Record actual start time on assigned core
        # This maintains the scheduling sequence Sk from Section III.B
        task.execution_unit_task_start_times[core] = start_time
        # Update core availability for next task
        # Core k cannot execute another task until current task finishes
        self.core_earliest_ready[core] = finish_time
        # Set task assignment (ki from Section II.B)
        # ki > 0 indicates local core execution
        task.assignment = core
        # Mark task as scheduled in initial scheduling phase
        task.is_scheduled = SchedulingState.SCHEDULED
        # Add task to execution sequence for this core
        # This implements Sk sequence tracking from Section III.B
        # Used later for task migration phase
        self.sequences[core].append(task.id)

    def calculate_cloud_phases_timing(self, task):
        """
        Calculates timing for the three-phase cloud execution model described in Section II.B.
        Implements timing calculations from equations (1), (2), (4), (5), and (6).
        """
        # Phase 1: RF Sending Phase
        # Ready time RTi^ws from equation (4) - when we can start sending
        send_ready = task.RT_ws
        # Finish time FTi^ws = RTi^ws + Ti^s 
        # Ti^s = datai/R^s from equation (1)
        # Time to send task specification and input data
        send_finish = send_ready + task.cloud_execution_times[0]
        # Phase 2: Cloud Computing Phase
        # Ready time RTi^c from equation (5)
        # Can start computing once sending is complete
        cloud_ready = send_finish
        # Finish time FTi^c = RTi^c + Ti^c
        # Ti^c is cloud computation time
        cloud_finish = cloud_ready + task.cloud_execution_times[1]
        # Phase 3: RF Receiving Phase
        # Ready time RTi^wr from equation (6)
        # Can start receiving once cloud computation finishes
        receive_ready = cloud_finish
        # Finish time FTi^wr considering:
        # 1. When results are ready (receive_ready)
        # 2. When wireless channel is available (wr_ready)
        # 3. Time to receive results Ti^r = data'i/R^r from equation (2)
        receive_finish = (
                max(self.wr_ready, receive_ready) +
                task.cloud_execution_times[2]
        )

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready,
                          receive_finish):
        """
        Schedules a task for cloud execution, updating all timing parameters and resource availability.
        Implements cloud scheduling from Section II.B and II.C.2 of the paper.
        """
        # Set timing parameters for three-phase cloud execution
        # Phase 1: RF Sending Phase
        task.RT_ws = send_ready  # When we can start sending (eq. 4)
        task.FT_ws = send_finish  # When sending completes (eq. 1)

        # Phase 2: Cloud Computing Phase
        task.RT_c = cloud_ready  # When cloud can start (eq. 5)
        task.FT_c = cloud_finish  # When cloud computation ends

        # Phase 3: RF Receiving Phase
        task.RT_wr = receive_ready  # When results are ready (eq. 6)
        task.FT_wr = receive_finish  # When results are received

        # Set overall execution finish time for precedence checking
        task.execution_finish_time = receive_finish

        # Clear local core finish time since executing on cloud
        # FTi^l = 0 indicates cloud execution as per Section II.C
        task.FT_l = 0

        # Initialize execution unit timing array
        # -1 indicates not scheduled on that unit
        task.execution_unit_task_start_times = [-1] * (self.k + 1)

        # Record cloud execution start time
        # Used for Sk sequence tracking in Section III.B
        task.execution_unit_task_start_times[self.k] = send_ready

        # Set task assignment (ki = 0 for cloud execution)
        # As specified in Section II.B
        task.assignment = self.k

        # Mark task as scheduled in initial phase
        task.is_scheduled = SchedulingState.SCHEDULED

        # Update wireless channel availability
        # Cannot send new task until current send completes
        self.ws_ready = send_finish
        # Cannot receive new results until current receive completes
        self.wr_ready = receive_finish

        # Add to cloud execution sequence
        # Maintains Sk sequences for task migration phase
        self.sequences[self.k].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """
        Schedules tasks with no predecessors (pred(vi) = ∅).
        Implements initial task scheduling from Section III.A.3 of the paper.
        Handles both local and cloud execution paths with appropriate ordering.
        """
        # Track tasks marked for cloud execution
        # These are scheduled after local tasks to enable pipeline staggering
        cloud_entry_tasks = []

        # First Phase: Schedule tasks assigned to local cores
        # Process local tasks first since they don't have sending dependencies
        for task in entry_tasks:
            if task.is_core_task:
                # Find optimal core assignment using criteria from Section III.A.3
                # Returns core k that minimizes finish time FTi^l
                core, start_time, finish_time = self.identify_optimal_local_core(task)

                # Schedule on chosen core
                # Updates task timing and core availability
                self.schedule_on_local_core(task, core, start_time, finish_time)
            else:
                # Collect cloud tasks for second phase
                cloud_entry_tasks.append(task)

        # Second Phase: Schedule cloud tasks
        # Process after local tasks to manage wireless channel congestion
        for task in cloud_entry_tasks:
            # Set wireless send ready time RTi^ws
            # Uses current wireless channel availability
            task.RT_ws = self.ws_ready

            # Calculate timing for three-phase cloud execution
            # Returns timing parameters for send, compute, and receive phases
            timing = self.calculate_cloud_phases_timing(task)

            # Schedule cloud execution
            # Updates task timing and wireless channel availability
            self.schedule_on_cloud(task, *timing)

    def calculate_non_entry_task_ready_times(self, task):
        """
        Calculates ready times for tasks that have predecessors.
        Implements equations (3) and (4) from Section II.C of the paper.
        """
        # Calculate local core ready time RTi^l (equation 3)
        # RTi^l = max(vj∈pred(vi)) max(FTj^l, FTj^wr)
        # Task can start on local core when all predecessors are complete:
        # - FTj^l: If predecessor executed locally
        # - FTj^wr: If predecessor executed on cloud
        task.RT_l = max(
            max(max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks),
            0  # Ensure non-negative ready time
        )

        # Calculate cloud sending ready time RTi^ws (equation 4)
        # RTi^ws = max(vj∈pred(vi)) max(FTj^l, FTj^ws)
        # Can start sending to cloud when:
        # 1. All predecessors have completed:
        #    - FTj^l: If predecessor executed locally
        #    - FTj^ws: If predecessor was sent to cloud
        # 2. Wireless sending channel is available
        task.RT_ws = max(
            max(max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks),
            self.ws_ready  # Channel availability
        )

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """
        Schedules tasks that have predecessors (pred(vi) ≠ ∅).
        Implements execution unit selection from Section III.A.3 of the paper.

        Makes scheduling decisions by:
        1. Calculating ready times based on predecessors
        2. Evaluating both local and cloud execution options
        3. Selecting execution path that minimizes finish time
        """
        # Process tasks in priority order (from task_prioritizing)
        for task in non_entry_tasks:
            # Calculate RTi^l and RTi^ws based on predecessor finish times
            # Implements equations (3) and (4)
            self.calculate_non_entry_task_ready_times(task)

            # If task was marked for cloud in primary assignment
            if not task.is_core_task:
                # Calculate three-phase cloud execution timing
                timing = self.calculate_cloud_phases_timing(task)
                # Schedule task on cloud
                self.schedule_on_cloud(task, *timing)
            else:
                # For tasks marked for local execution:
                # 1. Find best local core option
                core, start_time, finish_time = self.identify_optimal_local_core(
                    task, task.RT_l  # Consider ready time RTi^l
                )

                # 2. Calculate cloud execution option for comparison
                # "schedule task vi on the core or offload it to the cloud 
                # such that the finish time is minimized"
                timing = self.calculate_cloud_phases_timing(task)
                cloud_finish_time = timing[-1]  # FTi^wr

                # 3. Choose execution path with earlier finish time
                # This implements the minimum finish time criteria
                # from Section III.A.3
                if finish_time <= cloud_finish_time:
                    # Local execution is faster
                    self.schedule_on_local_core(task, core, start_time, finish_time)
                else:
                    # Cloud execution is faster
                    # Override primary assignment decision
                    task.is_core_task = False
                    self.schedule_on_cloud(task, *timing)


def execution_unit_selection(tasks):
    """
    Implements execution unit selection phase described in Section III.A.3.
    This is the third and final phase of the initial scheduling algorithm
    after primary assignment and task prioritizing.
    
    Args:
        tasks: List of tasks from the application graph G=(V,E)
    
    Returns:
        sequences: List of task sequences Sk for each execution unit
                  Used in task migration phase for energy optimization
    """
    # Initialize scheduler with tasks and K=3 cores
    # As described in Section II.B of the paper
    scheduler = InitialTaskScheduler(tasks, 3)

    # Order tasks by priority score from equation (15)
    # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
    # Higher priority indicates task is on critical path
    priority_orderered_tasks = scheduler.get_priority_ordered_tasks()

    # Classify tasks based on dependencies
    # Entry tasks: pred(vi) = ∅ (can start immediately)
    # Non-entry tasks: must wait for predecessors
    # Maintains priority ordering within each category
    entry_tasks, non_entry_tasks = scheduler.classify_entry_tasks(priority_orderered_tasks)

    # Two-phase scheduling process:
    # 1. Schedule entry tasks (no dependencies)
    #    - Process local tasks first
    #    - Then handle cloud tasks with pipeline staggering
    scheduler.schedule_entry_tasks(entry_tasks)

    # 2. Schedule non-entry tasks (with dependencies)
    #    - Calculate ready times based on predecessors
    #    - Compare local vs cloud execution times
    #    - Choose path that minimizes finish time
    scheduler.schedule_non_entry_tasks(non_entry_tasks)

    # Return task sequences for each execution unit
    # These Sk sequences are used in Section III.B
    # for the task migration algorithm
    return scheduler.sequences


def construct_sequence(tasks, task_id, execution_unit, original_sequence):
    """
   Implements the linear-time rescheduling algorithm described in Section III.B.2.
   Constructs new sequence after task migration while preserving task precedence.
   
   Args:
       tasks: List of all tasks in the application
       task_id: ID of task v_tar being migrated
       execution_unit: New execution location k_tar
       original_sequence: Current Sk sequences for all execution units
       
   Returns:
       Modified sequence sets after migrating the task
   """
    # Step 1: Create task lookup dictionary for O(1) access
    # Enables quick task object retrieval during sequence construction
    task_id_to_task = {task.id: task for task in tasks}

    # Step 2: Get the target task v_tar for migration
    target_task = task_id_to_task.get(task_id)

    # Step 3: Get ready time for insertion
    # RTi^l for local cores (k_tar > 0)
    # RTi^ws for cloud execution (k_tar = 0)
    target_task_rt = target_task.RT_l if target_task.is_core_task else target_task.RT_ws

    # Step 4: Remove task from original sequence
    # Implementation of equation (17):
    # "we will not change the ordering of tasks in the other cores"
    original_assignment = target_task.assignment
    original_sequence[original_assignment].remove(target_task.id)

    # Step 5: Get sequence for new execution unit
    # Prepare for ordered insertion based on start times
    new_sequence_task_list = original_sequence[execution_unit]

    # Get start times for tasks in new sequence
    # Used to maintain proper task ordering
    start_times = [
        task_id_to_task[task_id].execution_unit_task_start_times[execution_unit]
        for task_id in new_sequence_task_list
    ]

    # Step 6: Find insertion point using binary search
    # Implements "insert v_tar into S_k_tar such that v_tar is
    # executed after all its transitive predecessors and before
    # all its transitive successors"
    insertion_index = bisect.bisect_left(start_times, target_task_rt)

    # Step 7: Insert task at correct position
    # Maintains ordered sequence based on start times
    new_sequence_task_list.insert(insertion_index, target_task.id)

    # Step 8: Update task execution information
    # Set new assignment k_i and execution type
    target_task.assignment = execution_unit
    target_task.is_core_task = (execution_unit != 3)  # 3 indicates cloud

    return original_sequence


class KernelScheduler:
    """
    Implements the kernel (rescheduling) algorithm from Section III.B.2.
    Provides linear-time rescheduling for task migration phase of MCC scheduling.
    
    This scheduler maintains state for:
    - Task dependency tracking
    - Resource availability
    - Task sequencing
    """

    def __init__(self, tasks, sequences):
        """
        Initialize kernel scheduler for task migration rescheduling.
        """
        self.tasks = tasks
        # Sk sequences from equation (17)
        # sequences[k]: Tasks assigned to execution unit k
        # k = 0,1,2: Local cores
        # k = 3: Cloud execution
        self.sequences = sequences

        # Resource timing trackers
        # Track when each execution unit becomes available

        # RTi^l ready times for local cores (k > 0)
        # From equation (3): When each core can start next task
        self.RT_ls = [0] * 3  # Three cores

        # Ready times for cloud execution phases
        # [0]: RTi^ws - Wireless sending (eq. 4)
        # [1]: RTi^c  - Cloud computation (eq. 5)
        # [2]: RTi^wr - Result receiving (eq. 6)
        self.cloud_phases_ready_times = [0] * 3

        # Initialize task readiness tracking vectors
        # These implement the ready1 and ready2 vectors
        # described in Section III.B.2
        self.dependency_ready, self.sequence_ready = self.initialize_task_state()

    def initialize_task_state(self):
        """
        Initializes task readiness tracking vectors described in Section III.B.2.
        These vectors enable linear-time rescheduling by tracking:
        1. Task dependency completion (ready1)
        2. Sequence position readiness (ready2)
        """
        # Initialize ready1 vector (dependency tracking)
        # ready1[j] is number of immediate predecessors not yet scheduled
        # "ready1[j] is the number of immediate predecessors of task v[j]
        # that have not been scheduled"
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]

        # Initialize ready2 vector (sequence position tracking)
        # ready2[j] indicates if task is ready in its sequence:
        # -1: Task not in current sequence
        #  0: Task ready to execute (first in sequence or predecessor completed)
        #  1: Task waiting for predecessor in sequence
        sequence_ready = [-1] * len(self.tasks)

        # Process each execution sequence Sk
        for sequence in self.sequences:
            if sequence:  # Non-empty sequence
                # Mark first task in sequence as ready
                # "ready2[j] = 0 if all the tasks before task v[j] in
                # the same sequence have already been scheduled"
                sequence_ready[sequence[0] - 1] = 0

        return dependency_ready, sequence_ready

    def update_task_state(self, task):
        """
        Updates readiness vectors (ready1, ready2) for a task after scheduling changes.
        Implements the vector update logic from Section III.B.2 of the paper.
 
        This maintains the invariants required for linear-time scheduling:
        1. ready1[j] tracks unscheduled predecessors
        2. ready2[j] tracks sequence position readiness
        """
        # Only update state for unscheduled tasks
        # Once a task is KERNEL_SCHEDULED, its state is final
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update ready1 vector (dependency tracking)
            # "ready1[j] by one for all vj ∈ succ(vi)"
            # Count immediate predecessors that haven't been scheduled
            self.dependency_ready[task.id - 1] = sum(
                1 for pred_task in task.pred_tasks
                if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )

            # Update ready2 vector (sequence position)
            # Find task's position in its current execution sequence
            for sequence in self.sequences:
                if task.id in sequence:
                    idx = sequence.index(task.id)
                    if idx > 0:
                        # Task has predecessor in sequence
                        # Check if predecessor has been scheduled
                        prev_task = self.tasks[sequence[idx - 1] - 1]
                        self.sequence_ready[task.id - 1] = (
                            # 1: Waiting for predecessor
                            # 0: Predecessor completed
                            1 if prev_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
                            else 0
                        )
                    else:
                        # First task in sequence
                        # "ready2[j] = 0 if all the tasks before task vj
                        # in the same sequence have already been scheduled"
                        self.sequence_ready[task.id - 1] = 0
                    break

    def schedule_local_task(self, task):
        """
        Schedules a task for local core execution following Section II.C.1.
        Implements timing calculations for tasks executing on cores.

        Updates:
            - RTi^l: Local execution ready time (eq. 3)
            - FTi^l: Local execution finish time 
            - Core availability tracking
        """
        # Calculate ready time RTi^l for local execution
        # Implements equation (3): RTi^l = max(vj∈pred(vi)) max(FTj^l, FTj^wr)
        if not task.pred_tasks:
            # Entry tasks can start immediately
            task.RT_l = 0
        else:
            # Find latest completion time among predecessors
            # Consider both local (FTj^l) and cloud (FTj^wr) execution
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_wr)
                for pred_task in task.pred_tasks
            )
            task.RT_l = max(pred_task_completion_times, default=0)

        # Schedule on assigned core k
        core_index = task.assignment
        # Initialize execution timing array
        # Index 0-2: Local cores
        # Index 3: Cloud
        task.execution_unit_task_start_times = [-1] * 4

        # Calculate actual start time considering:
        # 1. Task ready time RTi^l
        # 2. Core availability (RT_ls[k])
        task.execution_unit_task_start_times[core_index] = max(
            self.RT_ls[core_index],  # Core availability
            task.RT_l  # Task ready time
        )

        # Calculate finish time FTi^l
        # FTi^l = start_time + Ti,k^l 
        # where Ti,k^l is execution time on core k
        task.FT_l = (
                task.execution_unit_task_start_times[core_index] +
                task.core_execution_times[core_index]
        )

        # Update core k's next available time
        self.RT_ls[core_index] = task.FT_l

        # Clear cloud execution timings
        # FTi^ws = FTi^c = FTi^wr = 0 for local tasks
        # As specified in Section II.C
        task.FT_ws = -1
        task.FT_c = -1
        task.FT_wr = -1

    def schedule_cloud_task(self, task):
        """
        Schedules three-phase cloud execution described in Section II.B.
        Implements timing calculations for:
        1. RF sending phase (equations 1, 4)
        2. Cloud computation phase (equation 5)
        3. RF receiving phase (equations 2, 6)

        Updates all cloud execution timing parameters and resource availability
        """
        # Calculate wireless sending ready time RTi^ws
        # Implements equation (4): RTi^ws = max(vj∈pred(vi)) max(FTj^l, FTj^ws)
        if not task.pred_tasks:
            # Entry tasks can start sending immediately
            task.RT_ws = 0
        else:
            # Find latest completion time among predecessors
            # Consider both local execution (FTj^l) and cloud sending (FTj^ws)
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_ws)
                for pred_task in task.pred_tasks
            )
            task.RT_ws = max(pred_task_completion_times)

        # Initialize timing array for execution units
        task.execution_unit_task_start_times = [-1] * 4
        # Set cloud start time considering:
        # 1. Wireless channel availability
        # 2. Task ready time RTi^ws
        task.execution_unit_task_start_times[3] = max(
            self.cloud_phases_ready_times[0],  # Channel availability
            task.RT_ws  # Task ready time
        )

        # Phase 1: RF Sending Phase
        # Implement equation (1): Ti^s = datai/R^s
        # Calculate finish time FTi^ws
        task.FT_ws = (
                task.execution_unit_task_start_times[3] +
                task.cloud_execution_times[0]  # Ti^s
        )
        # Update sending channel availability
        self.cloud_phases_ready_times[0] = task.FT_ws

        # Phase 2: Cloud Computing Phase
        # Implement equation (5): RTi^c calculation
        task.RT_c = max(
            task.FT_ws,  # Must finish sending
            max((pred_task.FT_c for pred_task in task.pred_tasks), default=0)
        )
        # Calculate cloud finish time FTi^c
        task.FT_c = (
                max(self.cloud_phases_ready_times[1], task.RT_c) +
                task.cloud_execution_times[1]  # Ti^c
        )
        # Update cloud availability
        self.cloud_phases_ready_times[1] = task.FT_c

        # Phase 3: RF Receiving Phase
        # Implement equation (6): RTi^wr = FTi^c
        task.RT_wr = task.FT_c
        # Calculate receiving finish time using equation (2)
        task.FT_wr = (
                max(self.cloud_phases_ready_times[2], task.RT_wr) +
                task.cloud_execution_times[2]  # Ti^r
        )
        # Update receiving channel availability
        self.cloud_phases_ready_times[2] = task.FT_wr

        # Clear local execution timing
        # FTi^l = 0 for cloud tasks as per Section II.C
        task.FT_l = -1

    def initialize_queue(self):
        """
        Initializes LIFO stack for linear-time scheduling described in Section III.B.2.
        Identifies initially ready tasks based on both dependency and sequence readiness.
        """
        # Create LIFO stack (implemented as deque)
        # A task vi is ready for scheduling when both:
        # 1. ready1[i] = 0: All predecessors scheduled
        # 2. ready2[i] = 0: Ready in execution sequence
        return deque(
            task for task in self.tasks
            if (
                # Check sequence readiness (ready2[i] = 0)
                # Task must be first in sequence or after scheduled task
                    self.sequence_ready[task.id - 1] == 0
                    and
                    # Check dependency readiness (ready1[i] = 0)
                    # All predecessors must be completely scheduled
                    all(pred_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED
                        for pred_task in task.pred_tasks)
            )
        )


def kernel_algorithm(tasks, sequences):
    """
   Implements the kernel (rescheduling) algorithm from Section III.B.2.
   Provides linear-time task rescheduling for the task migration phase.
   """
    # Initialize kernel scheduler with tasks and sequences
    # Handles timing calculations and readiness tracking
    scheduler = KernelScheduler(tasks, sequences)

    # Initialize LIFO stack with ready tasks
    # "initialized by pushing the task vi's with both
    # ready1[i] = 0 and ready2[i] = 0 into the empty stack"
    queue = scheduler.initialize_queue()

    # Main scheduling loop
    # "repeat the following steps until the stack becomes empty"
    while queue:
        # Pop next ready task from stack
        current_task = queue.popleft()
        # Mark as scheduled in kernel phase
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

        # Schedule based on execution type
        if current_task.is_core_task:
            # Schedule on assigned local core k
            # Updates RTi^l and FTi^l
            scheduler.schedule_local_task(current_task)
        else:
            # Schedule three-phase cloud execution
            # Updates RTi^ws, FTi^ws, RTi^c, FTi^c, RTi^wr, FTi^wr
            scheduler.schedule_cloud_task(current_task)

        # Update ready1 and ready2 vectors
        # "Update vectors ready1 (reducing ready1[j] by one for all
        # vj ∈ succ(vi)) and ready2, and push all the new tasks vj
        # with both ready1[j] = 0 and ready2[j] = 0 into the stack"
        for task in tasks:
            scheduler.update_task_state(task)

            # Add newly ready tasks to stack
            if (scheduler.dependency_ready[task.id - 1] == 0 and  # ready1[j] = 0
                    scheduler.sequence_ready[task.id - 1] == 0 and  # ready2[j] = 0
                    task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
                    task not in queue):
                queue.append(task)

    # Reset scheduling state for next iteration
    # Allows multiple runs of kernel algorithm during task migration
    for task in tasks:
        task.is_scheduled = SchedulingState.UNSCHEDULED

    return tasks


def generate_cache_key(tasks, task_idx, target_execution_unit):
    """
        Generates cache key for memoizing migration evaluations.
        Enables efficient evaluation of migration options in Section III.B.
        """
    # Create cache key from:
    # 1. Task being migrated (v_tar)
    # 2. Target execution unit (k_tar)
    # 3. Current task assignments (ki for all tasks)
    return (task_idx, target_execution_unit,
            tuple(task.assignment for task in tasks))


def evaluate_migration(tasks, seqs, task_idx, target_execution_unit, migration_cache, core_powers=[1, 2, 4],
                       cloud_sending_power=0.5):
    """
        Evaluates potential task migration scenario described in Section III.B.
        Uses caching to avoid redundant calculations.

        Args:
            tasks: List of all tasks
            seqs: Current Sk sequences
            task_idx: Index of task v_tar to migrate
            target_execution_unit: Proposed location k_tar
            migration_cache: Dictionary storing previous migration evaluations
            
        Returns:
            tuple: (T_total, E_total) after migration
            - T_total: Total completion time (eq. 10)
            - E_total: Total energy consumption (eq. 9)
        """
    # Generate cache key for this migration scenario
    cache_key = generate_cache_key(tasks, task_idx, target_execution_unit)

    # Check cache for previously evaluated scenario
    if cache_key in migration_cache:
        return migration_cache[cache_key]

    # Create copies to avoid modifying original state
    sequence_copy = [seq.copy() for seq in seqs]
    tasks_copy = deepcopy(tasks)

    # Apply migration and recalculate schedule
    sequence_copy = construct_sequence(
        tasks_copy,
        task_idx + 1,  # Convert to 1-based task ID
        target_execution_unit,
        sequence_copy
    )
    kernel_algorithm(tasks_copy, sequence_copy)

    # Calculate new metrics
    migration_T = total_time(tasks_copy)
    migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power)

    # Cache results
    migration_cache[cache_key] = (migration_T, migration_E)
    return migration_T, migration_E


def initialize_migration_choices(tasks):
    """
        Initializes possible migration choices for each task as described in Section III.B.
        """
    # Create matrix of migration possibilities:
    # N rows (tasks) x 4 columns (3 cores + cloud)
    # Implements "total of N × K migration choices"
    # from Section III.B outer loop
    migration_choices = np.zeros((len(tasks), 4), dtype=bool)

    # Set valid migration targets for each task
    for i, task in enumerate(tasks):
        if task.assignment == 3:
            # Cloud-assigned tasks (ki = 0)
            # Can potentially migrate to any local core
            migration_choices[i, :] = True
        else:
            # Locally-assigned tasks (ki > 0)
            # Can only migrate to current core or cloud
            # Maintains task's current valid execution options
            migration_choices[i, task.assignment] = True

    return migration_choices


def identify_optimal_migration(migration_trials_results, T_final, E_total, T_max):
    """
        Identifies optimal task migration as described in Section III.B.
        Implements two-step selection process for energy reduction while maintaining
        completion time constraints.
        """
    # Step 1: Find migrations that reduce energy without increasing time
    # "select the choice that results in the largest energy reduction
    # compared with the current schedule and no increase in T_total"
    best_energy_reduction = 0
    best_migration = None

    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Skip migrations violating T_max constraint
        if time > T_max:
            continue

        # Calculate potential energy reduction
        # ΔE = E_total_current - E_total_after
        energy_reduction = E_total - energy

        # Check if migration:
        # 1. Doesn't increase completion time (T_total)
        # 2. Reduces energy consumption (E_total)
        if time <= T_final and energy_reduction > 0:
            if energy_reduction > best_energy_reduction:
                best_energy_reduction = energy_reduction
                best_migration = (task_idx, resource_idx, time, energy)

    # Return best energy-reducing migration if found
    if best_migration:
        task_idx, resource_idx, time, energy = best_migration
        return TaskMigrationState(
            time=time,
            energy=energy,
            efficiency=best_energy_reduction,
            task_index=task_idx + 1,
            target_execution_unit=resource_idx + 1
        )

    # Step 2: If no direct energy reduction found
    # "select the one that results in the largest ratio of
    # energy reduction to the increase of T_total"
    migration_candidates = []
    for task_idx, resource_idx, time, energy in migration_trials_results:
        # Maintain T_max constraint
        if time > T_max:
            continue

        # Calculate energy reduction
        energy_reduction = E_total - energy
        if energy_reduction > 0:
            # Calculate efficiency ratio
            # ΔE / ΔT where ΔT is increase in completion time
            time_increase = max(0, time - T_final)
            if time_increase == 0:
                efficiency = float('inf')  # Prioritize no time increase
            else:
                efficiency = energy_reduction / time_increase

            heappush(migration_candidates,
                     (-efficiency, task_idx, resource_idx, time, energy))

    if not migration_candidates:
        return None

    # Return migration with best efficiency ratio
    neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
    return TaskMigrationState(
        time=T_best,
        energy=E_best,
        efficiency=-neg_ratio,
        task_index=n_best + 1,
        target_execution_unit=k_best + 1
    )


def optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5):
    """
   Implements the task migration algorithm from Section III.B of the paper.
   Optimizes energy consumption while maintaining completion time constraints.
   
   Args:
       tasks: List of tasks from application graph G=(V,E)
       sequence: Initial Sk sequences from minimal-delay scheduling
       T_final: Target completion time constraint T_max
       core_powers: Power consumption Pk for each core k
       cloud_sending_power: Power consumption P^s for RF sending
       
   Returns:
       tuple: (tasks, sequence) with optimized scheduling
   """
    # Convert core powers to numpy array for efficient operations
    core_powers = np.array(core_powers)

    # Cache for memoizing migration evaluations
    migration_cache = {}

    # Calculate initial energy consumption E_total (equation 9)
    current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)

    # Iterative improvement loop
    # "repeat the previous steps until the energy consumption
    # cannot be further minimized"
    energy_improved = True
    while energy_improved:
        # Store current energy for comparison
        previous_iteration_energy = current_iteration_energy

        # Get current schedule metrics
        current_time = total_time(tasks)  # T_total (equation 10)
        T_max = T_final * 1.5  # Allow some scheduling flexibility

        # Initialize migration possibilities matrix
        # N×K possible migrations as described in Section III.B
        migration_choices = initialize_migration_choices(tasks)

        # Evaluate all valid migration options
        migration_trials_results = []
        for task_idx in range(len(tasks)):
            for possible_execution_unit in range(4):
                if migration_choices[task_idx, possible_execution_unit]:
                    continue

                # Calculate T_total and E_total after migration
                migration_trial_time, migration_trial_energy = evaluate_migration(
                    tasks, sequence, task_idx, possible_execution_unit, migration_cache
                )
                migration_trials_results.append(
                    (task_idx, possible_execution_unit,
                     migration_trial_time, migration_trial_energy)
                )

        # Select best migration using two-step criteria
        # 1. Reduce energy without increasing time
        # 2. Best energy/time tradeoff ratio
        best_migration = identify_optimal_migration(
            migration_trials_results=migration_trials_results,
            T_final=current_time,
            E_total=previous_iteration_energy,
            T_max=T_max
        )

        # Exit if no valid migrations remain
        if best_migration is None:
            energy_improved = False
            break

        # Apply selected migration:
        # 1. Construct new sequences (Section III.B.2)
        sequence = construct_sequence(
            tasks,
            best_migration.task_index,
            best_migration.target_execution_unit - 1,
            sequence
        )

        # 2. Apply kernel algorithm for O(N) rescheduling
        kernel_algorithm(tasks, sequence)

        # Calculate new energy consumption
        current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)
        energy_improved = current_iteration_energy < previous_iteration_energy

        # Manage cache size for memory efficiency
        if len(migration_cache) > 1000:
            migration_cache.clear()

    return tasks, sequence


def print_task_schedule(tasks):
    """
    Prints formatted task scheduling information with optimized data handling.
    Shows execution timing for both local core and cloud execution paths.
    """
    # Pre-define constant mappings
    ASSIGNMENT_MAPPING = {
        0: "Core 1",
        1: "Core 2",
        2: "Core 3",
        3: "Cloud",
        -2: "Not Scheduled"
    }

    # Use list comprehension with conditional formatting
    schedule_data = []
    for task in tasks:
        base_info = {
            "Task ID": task.id,
            "Assignment": ASSIGNMENT_MAPPING.get(task.assignment, "Unknown")
        }

        if task.is_core_task:
            # Local core execution timing
            start_time = task.execution_unit_task_start_times[task.assignment]
            schedule_data.append({
                **base_info,
                "Execution Window": f"{start_time:.2f} → "
                                    f"{start_time + task.core_execution_times[task.assignment]:.2f}"
            })
        else:
            # Cloud execution phases timing
            send_start = task.execution_unit_task_start_times[3]
            send_end = send_start + task.cloud_execution_times[0]
            cloud_end = task.RT_c + task.cloud_execution_times[1]
            receive_end = task.RT_wr + task.cloud_execution_times[2]

            schedule_data.append({
                **base_info,
                "Send Phase": f"{send_start:.2f} → {send_end:.2f}",
                "Cloud Phase": f"{task.RT_c:.2f} → {cloud_end:.2f}",
                "Receive Phase": f"{task.RT_wr:.2f} → {receive_end:.2f}"
            })

    # Print formatted output
    print("\nTask Scheduling Details:")
    print("-" * 80)

    for entry in schedule_data:
        print("\n", end="")
        for key, value in entry.items():
            print(f"{key:15}: {value}")
        print("-" * 40)


def check_schedule_constraints(tasks):
    """
    Validates schedule constraints considering cloud task pipelining
    
    Args:
        tasks: List of Task objects with scheduling information
    Returns:
        tuple: (is_valid, violations)
    """
    violations = []

    def check_sending_channel():
        """Verify wireless sending channel is used sequentially"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.execution_unit_task_start_times[3])

        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            if current.FT_ws > next_task.execution_unit_task_start_times[3]:
                violations.append({
                    'type': 'Wireless Sending Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} sending ends at {current.FT_ws} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[3]}'
                })

    def check_computing_channel():
        """Verify cloud computing is sequential"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.RT_c)

        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            if current.FT_c > next_task.RT_c:
                violations.append({
                    'type': 'Cloud Computing Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} computing ends at {current.FT_c} but Task {next_task.id} starts at {next_task.RT_c}'
                })

    def check_receiving_channel():
        """Verify wireless receiving channel is sequential"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.RT_wr)

        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            if current.FT_wr > next_task.RT_wr:
                violations.append({
                    'type': 'Wireless Receiving Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} receiving ends at {current.FT_wr} but Task {next_task.id} starts at {next_task.RT_wr}'
                })

    def check_pipelined_dependencies():
        """Verify dependencies considering pipelined execution"""
        for task in tasks:
            if not task.is_core_task:  # For cloud tasks
                # Check if all pred_tasks have completed necessary phases
                for pred_task in task.pred_tasks:
                    if pred_task.is_core_task:
                        # Core pred_task must complete before child starts sending
                        if pred_task.FT_l > task.execution_unit_task_start_times[3]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Core Task {pred_task.id} finishes at {pred_task.FT_l} but Cloud Task {task.id} starts sending at {task.execution_unit_task_start_times[3]}'
                            })
                    else:
                        # Cloud pred_task must complete sending before child starts sending
                        if pred_task.FT_ws > task.execution_unit_task_start_times[3]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Parent Task {pred_task.id} sending phase ends at {pred_task.FT_ws} but Task {task.id} starts sending at {task.execution_unit_task_start_times[3]}'
                            })
            else:  # For core tasks
                # All pred_tasks must complete fully before core task starts
                for pred_task in task.pred_tasks:
                    pred_task_finish = (pred_task.FT_wr
                                        if not pred_task.is_core_task else pred_task.FT_l)
                    if pred_task_finish > task.execution_unit_task_start_times[task.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'pred_task': pred_task.id,
                            'child': task.id,
                            'detail': f'Parent Task {pred_task.id} finishes at {pred_task_finish} but Core Task {task.id} starts at {task.execution_unit_task_start_times[task.assignment]}'
                        })

    def check_core_execution():
        """Verify core tasks don't overlap"""
        core_tasks = [n for n in tasks if n.is_core_task]
        for core_id in range(3):  # Assuming 3 cores
            core_specific_tasks = [t for t in core_tasks if t.assignment == core_id]
            sorted_tasks = sorted(core_specific_tasks, key=lambda x: x.execution_unit_task_start_times[core_id])

            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]

                if current.FT_l > next_task.execution_unit_task_start_times[core_id]:
                    violations.append({
                        'type': f'Core {core_id} Execution Conflict',
                        'task1': current.id,
                        'task2': next_task.id,
                        'detail': f'Task {current.id} finishes at {current.FT_l} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[core_id]}'
                    })

    # Run all checks
    check_sending_channel()
    check_computing_channel()
    check_receiving_channel()
    check_pipelined_dependencies()
    check_core_execution()

    return len(violations) == 0, violations


"""
Pipeline Structure:
Each cloud task follows the three-phase execution described in Section II.B
"If task vi is offloaded onto the cloud, there are three phases in sequence:
(i) the RF sending phase,
(ii) the cloud computing phase, and
(iii) the RF receiving phase."
"The local core in the mobile device or the wireless sending channel can only process or send one task at a time, and preemption is not allowed in this framework. On the other hand, the cloud can execute a large number of tasks in parallel as long as there is no dependency among the tasks."
Task Precedence (Section II.C):


For wireless sending (RTws_i): A task can start sending only after all its predecessors have completed their relevant phases
For cloud computation (RTc_i): A task can start computing after:

It has completed its sending phase
All its cloud-executed predecessors have finished computing


For wireless receiving (RTwr_i): A task can start receiving immediately after cloud computation finishes

"""


def print_validation_report(tasks):
    """Print detailed schedule validation report"""
    is_valid, violations = check_schedule_constraints(tasks)

    print("\nSchedule Validation Report")
    print("=" * 50)

    if is_valid:
        print("Schedule is valid with all pipelining constraints satisfied!")
    else:
        print("Found constraint violations:")
        for v in violations:
            print(f"\nViolation Type: {v['type']}")
            print(f"Detail: {v['detail']}")


def print_task_graph(tasks):
    for task in tasks:
        succ_task_ids = [child.id for child in task.succ_task]
        pred_task_ids = [pred_task.id for pred_task in task.pred_tasks]
        print(f"Task {task.id}:")
        print(f"  Parents: {pred_task_ids}")
        print(f"  Children: {succ_task_ids}")
        print()


if __name__ == '__main__':
    """
    task20 = Task(id=20, pred_tasks=None, succ_task=[])
    task19 = Task(id=19, pred_tasks=None, succ_task=[task20])
    task18 = Task(id=18, pred_tasks=None, succ_task=[task20])
    task17 = Task(id=17, pred_tasks=None, succ_task=[task20])
    task16 = Task(id=16, pred_tasks=None, succ_task=[task19])
    task15 = Task(id=15, pred_tasks=None, succ_task=[task19])
    task14 = Task(id=14, pred_tasks=None, succ_task=[task18, task19])
    task13 = Task(id=13, pred_tasks=None, succ_task=[task17, task18])
    task12 = Task(id=12, pred_tasks=None, succ_task=[task17])
    task11 = Task(id=11, pred_tasks=None, succ_task=[task15, task16])
    task10 = Task(id=10, pred_tasks=None, succ_task=[task11,task15])
    task9 = Task(id=9, pred_tasks=None, succ_task=[task13,task14])
    task8 = Task(id=8, pred_tasks=None, succ_task=[task12,task13])
    task7 = Task(id=7, pred_tasks=None, succ_task=[task12])
    task6 = Task(id=6, pred_tasks=None, succ_task=[task10,task11])
    task5 = Task(id=5, pred_tasks=None, succ_task=[task9,task10])
    task4 = Task(id=4, pred_tasks=None, succ_task=[task8,task9])
    task3 = Task(id=3, pred_tasks=None, succ_task=[task7, task8])
    task2 = Task(id=2, pred_tasks=None, succ_task=[task7])
    task1 = Task(id=1, pred_tasks=None, succ_task=[task7])
    task1.pred_tasks = []
    task2.pred_tasks = []
    task3.pred_tasks = []
    task4.pred_tasks = []
    task5.pred_tasks = []
    task6.pred_tasks = []
    task7.pred_tasks = [task1,task2,task3]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task4,task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15,task16]
    task20.pred_tasks = [task17, task18,task19]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,task11,task12,task13,task14,task15,task16,task17,task18,task19,task20]

    task10 = Task(id=10, pred_tasks=None, succ_task=[])
    task9 = Task(id=9, pred_tasks=None, succ_task=[task10])
    task8 = Task(id=8, pred_tasks=None, succ_task=[task9])
    task7 = Task(id=7, pred_tasks=None, succ_task=[task9,task10])
    task6 = Task(id=6, pred_tasks=None, succ_task=[task10])
    task5 = Task(id=5, pred_tasks=None, succ_task=[task6])
    task4 = Task(id=4, pred_tasks=None, succ_task=[task7,task8])
    task3 = Task(id=3, pred_tasks=None, succ_task=[task7, task8])
    task2 = Task(id=2, pred_tasks=None, succ_task=[task5,task7])
    task1 = Task(id=1, pred_tasks=None, succ_task=[task2, task3, task4])
    task1.pred_tasks = []
    task2.pred_tasks = [task1]
    task3.pred_tasks = [task1]
    task4.pred_tasks = [task1]
    task5.pred_tasks = [task2]
    task6.pred_tasks = [task5]
    task7.pred_tasks = [task2,task3,task4]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task7,task8]
    task10.pred_tasks = [task6, task7, task9]
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    task20 = Task(id=20, pred_tasks=None, succ_task=[])
    task19 = Task(id=19, pred_tasks=None, succ_task=[])
    task18 = Task(id=18, pred_tasks=None, succ_task=[])
    task17 = Task(id=17, pred_tasks=None, succ_task=[])
    task16 = Task(id=16, pred_tasks=None, succ_task=[task19])
    task15 = Task(id=15, pred_tasks=None, succ_task=[task19])
    task14 = Task(id=14, pred_tasks=None, succ_task=[task18, task19])
    task13 = Task(id=13, pred_tasks=None, succ_task=[task17, task18])
    task12 = Task(id=12, pred_tasks=None, succ_task=[task17])
    task11 = Task(id=11, pred_tasks=None, succ_task=[task15, task16])
    task10 = Task(id=10, pred_tasks=None, succ_task=[task11,task15])
    task9 = Task(id=9, pred_tasks=None, succ_task=[task13,task14])
    task8 = Task(id=8, pred_tasks=None, succ_task=[task12,task13])
    task7 = Task(id=7, pred_tasks=None, succ_task=[task12])
    task6 = Task(id=6, pred_tasks=None, succ_task=[task10,task11])
    task5 = Task(id=5, pred_tasks=None, succ_task=[task9,task10])
    task4 = Task(id=4, pred_tasks=None, succ_task=[task8,task9])
    task3 = Task(id=3, pred_tasks=None, succ_task=[task7, task8])
    task2 = Task(id=2, pred_tasks=None, succ_task=[task7,task8])
    task1 = Task(id=1, pred_tasks=None, succ_task=[task7])
    task1.pred_tasks = []
    task2.pred_tasks = []
    task3.pred_tasks = []
    task4.pred_tasks = []
    task5.pred_tasks = []
    task6.pred_tasks = []
    task7.pred_tasks = [task1,task2,task3]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task4,task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15,task16]
    task20.pred_tasks = [task12]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,task11,task12,task13,task14,task15,task16,task17,task18,task19,task20]
    """

    task10 = Task(10)
    task9 = Task(9, succ_task=[task10])
    task8 = Task(8, succ_task=[task10])
    task7 = Task(7, succ_task=[task10])
    task6 = Task(6, succ_task=[task8])
    task5 = Task(5, succ_task=[task9])
    task4 = Task(4, succ_task=[task8, task9])
    task3 = Task(3, succ_task=[task7])
    task2 = Task(2, succ_task=[task8, task9])
    task1 = Task(1, succ_task=[task2, task3, task4, task5, task6])
    task10.pred_tasks = [task7, task8, task9]
    task9.pred_tasks = [task2, task4, task5]
    task8.pred_tasks = [task2, task4, task6]
    task7.pred_tasks = [task3]
    task6.pred_tasks = [task1]
    task5.pred_tasks = [task1]
    task4.pred_tasks = [task1]
    task3.pred_tasks = [task1]
    task2.pred_tasks = [task1]
    task1.pred_tasks = []
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    print_task_graph(tasks)

    primary_assignment(tasks)
    task_prioritizing(tasks)
    sequence = execution_unit_selection(tasks)
    T_final = total_time(tasks)
    E_total = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("INITIAL SCHEDULING APPLICATION COMPLETION TIME: ", T_final)
    print("INITIAL APPLICATION ENERGY CONSUMPTION:", E_total)
    print("INITIAL TASK SCHEDULE: ")
    print_task_schedule(tasks)
    print_validation_report(tasks)
    # check_mcc_constraints(tasks)

    tasks2, sequence = optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    print("final sequence: ")
    for s in sequence:
        print([i for i in s])

    T_final = total_time(tasks)
    E_final = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("FINAL SCHEDULING APPLICATION COMPLETION TIME: ", T_final)
    print("FINAL APPLICATION ENERGY CONSUMPTION:", E_final)
    print("FINAL TASK SCHEDULE: ")
    print_task_schedule(tasks2)
    print_validation_report(tasks2)
    # check_mcc_constraints(tasks2)
