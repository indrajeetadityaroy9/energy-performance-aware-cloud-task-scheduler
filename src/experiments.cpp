#include "mcc.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kLowerDurationRatio = 0.5;
constexpr double kUpperDurationRatio = 1.5;

void check_mean(double mean, bool zero_allowed, const char* name) {
    const double lower = kLowerDurationRatio * mean;
    const double upper = kUpperDurationRatio * mean;
    if (!std::isfinite(mean) || mean < 0.0 || (!zero_allowed && mean == 0.0) ||
        !std::isfinite(lower) || !std::isfinite(upper) ||
        (!zero_allowed && lower == 0.0)) {
        throw std::invalid_argument(
            std::string(name) +
            " must define finite uniform duration bounds (positive except receive)");
    }
}

void check_generator_config(const GeneratorConfig& config) {
    const auto max_int = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (config.tasks == 0 || config.tasks > max_int || config.cores == 0 ||
        config.cores > max_int) {
        throw std::invalid_argument("Task/core counts must be in [1, INT_MAX]");
    }
    if (!std::isfinite(config.edge_density) || config.edge_density < 0.0 ||
        config.edge_density > 1.0) {
        throw std::invalid_argument("Density must be finite and in [0,1]");
    }
    if (!std::isfinite(config.speedup) || config.speedup < 1.0) {
        throw std::invalid_argument("Speedup beta must be finite and at least 1");
    }

    check_mean(config.local_mean, false, "Local mean");
    check_mean(config.send_mean, false, "Send mean");
    check_mean(config.compute_mean, false, "Compute mean");
    check_mean(config.receive_mean, true, "Receive mean");

    // Validate the fastest core's lower bound before allocating or sampling.
    double fastest_lower_bound = kLowerDurationRatio * config.local_mean;
    for (std::size_t core = 1; core < config.cores; ++core) {
        fastest_lower_bound /= config.speedup;
        if (fastest_lower_bound == 0.0) {
            throw std::invalid_argument("Core speedup underflows local durations");
        }
    }
}

std::size_t expected_degree(double density, std::size_t possible_edges) {
    return static_cast<std::size_t>(
        std::ceil(density * static_cast<double>(possible_edges)));
}

void reserve_dependency_storage(std::vector<Task>& graph, double density) {
    const std::size_t task_count = graph.size();
    for (std::size_t index = 0; index < task_count; ++index) {
        graph[index].pred_tasks.reserve(expected_degree(density, index));
        graph[index].succ_tasks.reserve(
            expected_degree(density, task_count - index - 1));
    }
}

EnergyModel validated_energy_model(EnergyModel energy, std::size_t core_count_value) {
    if (energy.core_powers.empty()) {
        energy.core_powers = default_core_powers(core_count_value);
    }
    if (energy.core_powers.size() != core_count_value ||
        !std::isfinite(energy.sending_power) || energy.sending_power < 0.0) {
        throw std::invalid_argument(
            "Energy model needs K finite nonnegative core powers and RF power");
    }
    for (double power : energy.core_powers) {
        if (!std::isfinite(power) || power < 0.0) {
            throw std::invalid_argument("Invalid core power");
        }
    }
    return energy;
}

bool improves_baseline(const BaselineResult& result, bool feasible, double energy,
                       Time time, double best_energy, Time best_time) {
    if (feasible) {
        return !result.feasible || energy < best_energy ||
               (energy == best_energy && time < best_time);
    }
    return !result.feasible && (result.tasks.empty() || time < best_time);
}

}

// Section IV task-graph generation

std::vector<Task> generate_task_graph(const GeneratorConfig& config) {
    check_generator_config(config);

    std::mt19937_64 random(config.seed);
    std::uniform_real_distribution<Time> local_duration(
        kLowerDurationRatio * config.local_mean,
        kUpperDurationRatio * config.local_mean);
    std::uniform_real_distribution<Time> send_duration(
        kLowerDurationRatio * config.send_mean,
        kUpperDurationRatio * config.send_mean);
    std::uniform_real_distribution<Time> compute_duration(
        kLowerDurationRatio * config.compute_mean,
        kUpperDurationRatio * config.compute_mean);
    std::uniform_real_distribution<Time> receive_duration(
        kLowerDurationRatio * config.receive_mean,
        kUpperDurationRatio * config.receive_mean);

    std::vector<Task> graph;
    graph.reserve(config.tasks);
    for (std::size_t index = 0; index < config.tasks; ++index) {
        std::vector<Time> local(config.cores);
        local.front() = local_duration(random);
        for (std::size_t core = 1; core < config.cores; ++core) {
            local[core] = local[core - 1] / config.speedup;
        }

        const std::array<Time, 3> remote = {
            send_duration(random), compute_duration(random), receive_duration(random)};
        graph.emplace_back(static_cast<int>(index + 1), std::move(local), remote);
    }

    reserve_dependency_storage(graph, config.edge_density);
    std::bernoulli_distribution select_edge(config.edge_density);
    for (std::size_t source = 0; source < graph.size(); ++source) {
        for (std::size_t destination = source + 1; destination < graph.size();
             ++destination) {
            if (select_edge(random)) {
                graph[source].succ_tasks.push_back(graph[destination].id);
                graph[destination].pred_tasks.push_back(graph[source].id);
            }
        }
    }

    validate_task_graph(graph);
    return graph;
}

// Baseline 1: random fixed assignments

BaselineResult random_assignment_baseline(const std::vector<Task>& graph, Time deadline, EnergyModel energy, std::size_t trials, std::uint64_t seed) {
    validate_task_graph(graph);
    const std::size_t cores = core_count(graph);
    if (cores > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Too many cores for assignment indices");
    }
    if (!std::isfinite(deadline) || deadline < 0.0 || trials == 0) {
        throw std::invalid_argument(
            "Baseline needs a finite nonnegative deadline and at least one trial");
    }
    energy = validated_energy_model(std::move(energy), cores);

    std::mt19937_64 random(seed);
    std::uniform_int_distribution<int> select_assignment(0, static_cast<int>(cores));

    BaselineResult result;
    double best_energy = std::numeric_limits<double>::infinity();
    Time best_time = std::numeric_limits<Time>::infinity();

    // The scheduler resets all mutable timing fields on every invocation, so the
    // task and assignment buffers can be reused instead of deep-copied per trial.
    std::vector<Task> tasks = graph;
    std::vector<int> assignments(tasks.size());
    for (std::size_t trial = 0; trial < trials; ++trial) {
        for (int& resource : assignments) {
            resource = select_assignment(random);
        }

        Sequences sequences = fixed_assignment_schedule(tasks, assignments);
        const auto [valid, errors] = validate_schedule_constraints(
            tasks, std::numeric_limits<Time>::infinity(), &sequences);
        if (!valid) {
            const std::string detail =
                errors.empty() ? "unspecified validation failure" : errors.front();
            throw std::runtime_error("Baseline produced an invalid schedule: " +
                                     detail);
        }

        const Time time = total_time(tasks);
        const double cost = total_energy(tasks, energy.core_powers,
                                         energy.sending_power);
        if (!std::isfinite(cost)) {
            throw std::overflow_error("Baseline energy overflow");
        }

        const bool feasible = time <= deadline;
        ++result.trials;
        if (improves_baseline(result, feasible, cost, time, best_energy,
                              best_time)) {
            result.tasks = tasks;
            result.sequences = std::move(sequences);
            result.feasible = feasible;
            best_time = time;
            best_energy = cost;
        }
    }

    // If no trial met the deadline, return the fastest sampled failed schedule.
    // This is not evidence that no feasible assignment exists.
    return result;
}
