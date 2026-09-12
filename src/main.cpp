#include "mcc.hpp"

#include <array>
#include <bitset>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<Time, 5> example_deadlines = {27, 37.5, 51, 49.5, 43.5};

struct Options {
    GeneratorConfig generator;
    EnergyModel energy;
    std::size_t trials = 10000;
    int graph = 0;
    Time deadline = 100;
    bool experiment = false;
    bool local_only = false;
    bool custom_deadline = false;
};

enum class Option : std::uint8_t {
    experiment,
    local_only,
    graph,
    deadline,
    powers,
    rf_power,
    tasks,
    cores,
    density,
    seed,
    trials,
    local_mean,
    send_mean,
    compute_mean,
    receive_mean,
    speedup,
    count,
};

struct OptionSpec {
    std::string_view name;
    Option id;
    bool takes_value;
    bool generator_option;
};

constexpr std::size_t option_count = static_cast<std::size_t>(Option::count);
using SeenOptions = std::bitset<option_count>;

constexpr std::array<OptionSpec, option_count> option_specs = {{
    {"--experiment", Option::experiment, false, false},
    {"--local-only", Option::local_only, false, false},
    {"--graph", Option::graph, true, false},
    {"--deadline", Option::deadline, true, false},
    {"--powers", Option::powers, true, false},
    {"--rf-power", Option::rf_power, true, false},
    {"--tasks", Option::tasks, true, true},
    {"--cores", Option::cores, true, true},
    {"--density", Option::density, true, true},
    {"--seed", Option::seed, true, true},
    {"--trials", Option::trials, true, true},
    {"--local-mean", Option::local_mean, true, true},
    {"--send-mean", Option::send_mean, true, true},
    {"--compute-mean", Option::compute_mean, true, true},
    {"--receive-mean", Option::receive_mean, true, true},
    {"--speedup", Option::speedup, true, true},
}};

constexpr std::size_t option_index(Option option) {
    return static_cast<std::size_t>(option);
}

const OptionSpec* find_option(std::string_view name) {
    for (const auto& option : option_specs) {
        if (option.name == name) {
            return &option;
        }
    }
    return nullptr;
}

double parse_real(const std::string& value) {
    if (value.empty() || value.find_first_of(" \t\r\n\f\v") != std::string::npos) {
        throw std::invalid_argument("Expected a finite real, got: " + value);
    }

    std::size_t used = 0;
    const double result = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument("Expected a finite real, got: " + value);
    }
    return result;
}

std::uint64_t parse_integer(const std::string& value) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("Expected an unsigned integer, got: " + value);
    }
    return result;
}

std::size_t parse_count(const std::string& value) {
    const auto result = parse_integer(value);
    if (result == 0 || result > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("Count must be positive and fit size_t");
    }
    return static_cast<std::size_t>(result);
}

std::vector<double> parse_powers(const std::string& value) {
    if (value.empty() || value.back() == ',') {
        throw std::invalid_argument("Empty power entry");
    }

    std::size_t entry_count = 1;
    for (const char character : value) {
        entry_count += character == ',';
    }

    std::vector<double> powers;
    powers.reserve(entry_count);

    std::size_t begin = 0;
    while (begin < value.size()) {
        const std::size_t separator = value.find(',', begin);
        const std::size_t length = separator == std::string::npos
                                       ? value.size() - begin
                                       : separator - begin;
        const std::string entry = value.substr(begin, length);
        const double power = parse_real(entry);
        if (power < 0) {
            throw std::invalid_argument("Core powers must be nonnegative");
        }
        powers.push_back(power);

        if (separator == std::string::npos) {
            break;
        }
        begin = separator + 1;
    }
    return powers;
}

void apply_option(Options& options, Option option, const std::string& value) {
    switch (option) {
        case Option::experiment:
            options.experiment = true;
            return;
        case Option::local_only:
            options.local_only = true;
            return;
        case Option::graph: {
            const auto graph = parse_integer(value);
            if (graph < 1 || graph > example_deadlines.size()) {
                throw std::invalid_argument("Graph must be 1..5");
            }
            options.graph = static_cast<int>(graph);
            return;
        }
        case Option::deadline:
            options.deadline = parse_real(value);
            options.custom_deadline = true;
            if (options.deadline < 0) {
                throw std::invalid_argument("Deadline must be nonnegative");
            }
            return;
        case Option::powers:
            options.energy.core_powers = parse_powers(value);
            return;
        case Option::rf_power:
            options.energy.sending_power = parse_real(value);
            if (options.energy.sending_power < 0) {
                throw std::invalid_argument("RF power must be nonnegative");
            }
            return;
        case Option::tasks:
            options.generator.tasks = parse_count(value);
            return;
        case Option::cores:
            options.generator.cores = parse_count(value);
            return;
        case Option::density:
            options.generator.edge_density = parse_real(value);
            return;
        case Option::seed:
            options.generator.seed = parse_integer(value);
            return;
        case Option::trials:
            options.trials = parse_count(value);
            return;
        case Option::local_mean:
            options.generator.local_mean = parse_real(value);
            return;
        case Option::send_mean:
            options.generator.send_mean = parse_real(value);
            return;
        case Option::compute_mean:
            options.generator.compute_mean = parse_real(value);
            return;
        case Option::receive_mean:
            options.generator.receive_mean = parse_real(value);
            return;
        case Option::speedup:
            options.generator.speedup = parse_real(value);
            return;
        case Option::count:
            break;
    }
    throw std::logic_error("Invalid option identifier");
}

void validate_options(const Options& options, const SeenOptions& seen,
                      bool has_generator_option) {
    if (has_generator_option && !options.experiment) {
        throw std::invalid_argument("Generator/trial options require --experiment");
    }
    if (options.experiment && (options.graph != 0 || options.local_only)) {
        throw std::invalid_argument(
            "--experiment cannot be combined with --graph or --local-only");
    }
    if (!options.experiment && options.custom_deadline && options.graph == 0) {
        throw std::invalid_argument("Example --deadline requires --graph");
    }
    if (options.local_only && seen.test(option_index(Option::rf_power))) {
        throw std::invalid_argument("--rf-power is inappropriate with --local-only");
    }

    const std::size_t core_count = options.experiment ? options.generator.cores : 3;
    if (!options.energy.core_powers.empty() &&
        options.energy.core_powers.size() != core_count) {
        throw std::invalid_argument("--powers must contain exactly K entries");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    SeenOptions seen;
    bool has_generator_option = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view name = argv[index];
        const OptionSpec* spec = find_option(name);
        if (spec == nullptr) {
            throw std::invalid_argument("Unknown option: " + std::string(name));
        }

        const std::size_t slot = option_index(spec->id);
        if (seen.test(slot)) {
            throw std::invalid_argument("Duplicate option: " + std::string(name));
        }
        seen.set(slot);

        std::string value;
        if (spec->takes_value) {
            if (++index == argc) {
                throw std::invalid_argument("Missing value for " + std::string(name));
            }
            value = argv[index];
        }

        has_generator_option = has_generator_option || spec->generator_option;
        apply_option(options, spec->id, value);
    }

    validate_options(options, seen, has_generator_option);
    return options;
}

std::vector<double> resolved_powers(const EnergyModel& energy, std::size_t cores) {
    return energy.core_powers.empty() ? default_core_powers(cores) : energy.core_powers;
}

bool report_schedule(const std::string& name, const std::vector<Task>& tasks,
                     const Sequences& sequences, bool feasible, Time deadline,
                     const EnergyModel& energy, double runtime_seconds) {
    std::cout << name << ": feasible=" << (feasible ? "yes" : "no")
              << " runtime_seconds=" << runtime_seconds;

    if (tasks.empty()) {
        std::cout << " no schedule available\n";
        throw std::runtime_error(name + ": unexpectedly empty schedule");
    }

    const auto [structural_valid, structural_errors] = validate_schedule_constraints(
        tasks, std::numeric_limits<Time>::infinity(), &sequences);
    if (!structural_valid) {
        std::cout << '\n';
        for (const auto& error : structural_errors) {
            std::cout << "  FAIL: " << error << '\n';
        }
        throw std::runtime_error(name + ": invalid schedule constraints");
    }

    const Time completion_time = total_time(tasks);
    if (feasible != (completion_time <= deadline)) {
        throw std::runtime_error(
            name + ": feasibility flag disagrees with exact deadline comparison");
    }

    const auto powers = resolved_powers(energy, core_count(tasks));
    const double energy_cost = total_energy(tasks, powers, energy.sending_power);
    if (!std::isfinite(energy_cost)) {
        throw std::overflow_error("Reported energy overflow");
    }

    std::cout << " time=" << completion_time << " energy=" << energy_cost
              << " deadline=" << deadline << '\n';

    const auto [valid, errors] = validate_schedule_constraints(tasks, deadline, &sequences);
    for (const auto& error : errors) {
        std::cout << "  FAIL: " << error << '\n';
    }
    if (valid != feasible) {
        throw std::runtime_error(
            name + ": deadline validator disagrees with exact feasibility");
    }
    if (valid) {
        std::cout << "  PASS: schedule constraints\n";
    }
    return feasible;
}

bool run_scheduler(const std::string& name, const std::vector<Task>& graph, Time deadline,
                   const EnergyModel& energy, bool cloud_enabled, bool print_details) {
    const auto start = Clock::now();
    const auto result = schedule_application(graph, deadline, energy, cloud_enabled);
    const double runtime_seconds =
        std::chrono::duration<double>(Clock::now() - start).count();

    const auto [initial_valid, initial_errors] = validate_schedule_constraints(
        result.initial_tasks, std::numeric_limits<Time>::infinity(),
        &result.initial_sequences);
    for (const auto& error : initial_errors) {
        std::cout << "  FAIL initial: " << error << '\n';
    }
    if (!initial_valid) {
        throw std::runtime_error(name + ": invalid initial schedule constraints");
    }

    const bool final_valid = report_schedule(name, result.tasks, result.sequences,
                                             result.feasible, deadline, energy,
                                             runtime_seconds);

    const auto powers = resolved_powers(energy, core_count(graph));
    const double initial_energy =
        total_energy(result.initial_tasks, powers, energy.sending_power);
    if (!std::isfinite(initial_energy)) {
        throw std::overflow_error("Initial energy overflow");
    }

    std::cout << "  initial_time=" << total_time(result.initial_tasks)
              << " initial_energy=" << initial_energy
              << " accepted_migrations=" << result.migrations.accepted << '\n';

    if (!result.feasible) {
        std::cout << "  Initial heuristic schedule missed the deadline. "
                     "This is not proof no feasible solution exists.\n";
    }
    if (print_details) {
        print_schedule_tasks(result.tasks);
        print_schedule_sequences(result.sequences);
    }

    return final_valid;
}

void print_experiment_configuration(const Options& options) {
    const auto& generator = options.generator;
    std::cout
        << "Section IV-style comparison: uniform synthetic generator, NOT original random "
           "datasets.\n"
        << "Forward edges are independent Bernoulli(density). Durations uniform "
           "[0.5*mean,1.5*mean]; local core k uses base/beta^k.\n"
        << "tasks=" << generator.tasks << " cores=" << generator.cores
        << " density=" << generator.edge_density << " seed=" << generator.seed
        << " trials=" << options.trials << " deadline=" << options.deadline
        << " local_mean=" << generator.local_mean << " send_mean=" << generator.send_mean
        << " compute_mean=" << generator.compute_mean
        << " receive_mean=" << generator.receive_mean
        << " beta=" << generator.speedup << '\n';
}

bool run_experiment(const Options& options) {
    const auto graph = generate_task_graph(options.generator);
    print_experiment_configuration(options);

    bool success = run_scheduler("Proposed", graph, options.deadline, options.energy, true,
                                 false);

    const auto start = Clock::now();
    const auto baseline = random_assignment_baseline(
        graph, options.deadline, options.energy, options.trials, options.generator.seed);
    const double runtime_seconds =
        std::chrono::duration<double>(Clock::now() - start).count();

    const bool baseline_valid = report_schedule(
        "Baseline1 (minimum-energy feasible random fixed assignment)", baseline.tasks,
        baseline.sequences, baseline.feasible, options.deadline, options.energy,
        runtime_seconds);
    success = baseline_valid && success;

    if (!baseline.feasible) {
        std::cout << "  No sampled assignment met deadline; showing fastest failed trial, "
                     "not proof of impossibility.\n";
    }

    const bool local_valid = run_scheduler(
        "Baseline2 (local-only scheduling and energy optimization)", graph,
        options.deadline, options.energy, false, false);
    return local_valid && success;
}

bool run_examples(const Options& options) {
    const auto graphs = example_graphs();
    if (graphs.size() != example_deadlines.size()) {
        throw std::runtime_error("Expected five example graphs");
    }

    bool success = true;
    for (std::size_t index = 0; index < graphs.size(); ++index) {
        if (options.graph != 0 && static_cast<std::size_t>(options.graph) != index + 1) {
            continue;
        }

        const Time deadline =
            options.custom_deadline ? options.deadline : example_deadlines[index];
        const bool valid = run_scheduler(
            "Example " + std::to_string(index + 1), graphs[index], deadline,
            options.energy, !options.local_only, true);
        success = valid && success;
    }
    return success;
}

int run(const Options& options) {
    std::cout << std::setprecision(10);
    const bool success = options.experiment ? run_experiment(options) : run_examples(options);
    return success ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 2;
    }
}
