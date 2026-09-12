#!/bin/sh

set -eu

bin=${1:-bin/mcc_scheduler}
checks=0
output=

fail() {
    printf 'FAIL: %s\n%s\n' "$1" "$output" >&2
    exit 1
}

if [ ! -x "$bin" ]; then
    fail "scheduler binary is not executable: $bin"
fi

expect_exit() {
    expected_exit=$1
    shift

    actual_exit=0
    output=$("$bin" "$@" 2>&1) || actual_exit=$?
    if [ "$actual_exit" -ne "$expected_exit" ]; then
        fail "expected exit $expected_exit, got $actual_exit: $*"
    fi
    checks=$((checks + 1))
}

contains() {
    expected_text=$1
    case $output in
        *"$expected_text"*) ;;
        *) fail "output missing $expected_text" ;;
    esac
    checks=$((checks + 1))
}

# Default and documented example workflows ----------------------------------

expect_exit 2 --help
contains 'Unknown option: --help'

expect_exit 0
contains 'Example 5: feasible=yes'

expect_exit 0 --graph 1 --deadline 27
contains 'time=26'
contains 'PASS: schedule constraints'

expect_exit 0 --graph 1 --deadline 100 --local-only
contains 'feasible=yes'

expect_exit 0 --graph 1 --deadline 100 --powers 1,2,4 --rf-power 100
contains 'feasible=yes'

expect_exit 1 --graph 1 --deadline 0
contains 'not proof no feasible solution exists'

# Section IV experiment workflows

expect_exit 0 --experiment --tasks 11 --cores 6 --seed 42 \
    --deadline 1000 --trials 100
contains 'Proposed: feasible=yes'
contains 'Baseline1'
contains 'Baseline2'
contains 'cores=6'

expect_exit 0 --experiment --tasks 4 --cores 1 --density 0 --seed 7 \
    --deadline 1000 --trials 20 --receive-mean 0
contains 'feasible=yes'

expect_exit 1 --experiment --tasks 3 --deadline 0 --trials 10
contains 'No sampled assignment met deadline'

# Parsing, range, duplicate, and mode validation -----------------------------

expect_exit 2 --graph 0
expect_exit 2 --graph 6
expect_exit 2 --graph 1 --deadline nan
expect_exit 2 --graph 1 --deadline 10junk
expect_exit 2 --graph 1 --deadline -1
expect_exit 2 --deadline 27
expect_exit 2 --graph
expect_exit 2 --unknown
expect_exit 2 --graph 1 --graph 2
expect_exit 2 --help --graph 1
expect_exit 2 --cores 6
expect_exit 2 --experiment --graph 1
expect_exit 2 --experiment --local-only
expect_exit 2 --experiment --trials 0
expect_exit 2 --experiment --tasks -1
expect_exit 2 --experiment --density 1.1
expect_exit 2 --experiment --send-mean 0
expect_exit 2 --experiment --speedup 0.5
expect_exit 2 --experiment --cores 6 --powers 1,2,4
expect_exit 2 --powers 1,,4
expect_exit 2 --powers 1,2,4,
expect_exit 2 --local-only --rf-power 1

if [ "$checks" -ne 44 ]; then
    fail "acceptance suite executed $checks checks instead of 44"
fi
printf 'PASS: %s CLI acceptance checks\n' "$checks"
