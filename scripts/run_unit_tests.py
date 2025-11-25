"""Standalone unit test runner that works around import issues."""

import sys
from pathlib import Path

# Add src to path and change to src directory
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import os
os.chdir(src_path)

print("=" * 70)
print("RUNNING UNIT TESTS FOR AGENT SWARM - PRP-02")
print("=" * 70)
print()

# Now imports should work
from core.state import (
    create_initial_state,
    validate_state,
    update_state,
    serialize_state,
    deserialize_state,
    get_state_summary,
)
from core.edges import (
    route_from_coordinator,
    route_from_tester,
    route_from_validator,
    should_retry,
    needs_human_approval,
)
from models.state_models import WorkflowStatus, AgentRole

import asyncio

# Test counter
tests_run = 0
tests_passed = 0
tests_failed = 0

def test(name):
    """Decorator for tests."""
    def decorator(func):
        def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
                tests_passed += 1
                print(f"  [PASS] {name}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"  [FAIL] {name}: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"  [ERROR] {name}: {e}")
                return False
        return wrapper
    return decorator


print("TEST SUITE 1: State Management")
print("-" * 70)

@test("Create initial state")
def test_create_initial_state():
    task = {"id": "test_1", "description": "Test task"}
    state = create_initial_state(task=task, project_id="test_project")
    assert state["workflow_id"] is not None
    assert state["project_id"] == "test_project"
    assert state["status"] == WorkflowStatus.PENDING.value

test_create_initial_state()

@test("Validate state - valid")
def test_validate_state_valid():
    task = {"id": "test_1", "description": "Test"}
    state = create_initial_state(task=task, project_id="test_project")
    assert validate_state(state) is True

test_validate_state_valid()

@test("Validate state - invalid")
def test_validate_state_invalid():
    invalid_state = {"workflow_id": "test"}
    assert validate_state(invalid_state) is False

test_validate_state_invalid()

@test("Update state immutably")
def test_update_state():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    original_version = state["state_version"]

    updated = update_state(state, {"status": WorkflowStatus.PLANNING.value})

    assert state["state_version"] == original_version  # Original unchanged
    assert updated["state_version"] == original_version + 1
    assert updated["status"] == WorkflowStatus.PLANNING.value

test_update_state()

@test("Serialize and deserialize state")
def test_serialize_deserialize():
    task = {"id": "test", "description": "Test"}
    original = create_initial_state(task=task, project_id="test")

    serialized = serialize_state(original)
    deserialized = deserialize_state(serialized)

    assert deserialized["workflow_id"] == original["workflow_id"]
    assert deserialized["project_id"] == original["project_id"]

test_serialize_deserialize()

@test("Get state summary")
def test_get_state_summary():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    summary = get_state_summary(state)

    assert "workflow_id" in summary
    assert "status" in summary

test_get_state_summary()

print()
print("TEST SUITE 2: Workflow Routing")
print("-" * 70)

@test("Route from coordinator - pending to planner")
def test_route_coordinator_pending():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    next_node = route_from_coordinator(state)
    assert next_node == "planner"

test_route_coordinator_pending()

@test("Route from coordinator - complete to end")
def test_route_coordinator_complete():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"status": WorkflowStatus.COMPLETE.value})
    next_node = route_from_coordinator(state)
    assert next_node == "end"

test_route_coordinator_complete()

@test("Route from tester - passing tests")
def test_route_tester_pass():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"test_results": {"all_passed": True}})
    next_node = route_from_tester(state)
    assert next_node == "validator"

test_route_tester_pass()

@test("Route from tester - failing tests")
def test_route_tester_fail():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"test_results": {"all_passed": False}})
    next_node = route_from_tester(state)
    assert next_node == "debugger"

test_route_tester_fail()

@test("Route from validator - approved")
def test_route_validator_approved():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"validation_results": {"approved": True}})
    next_node = route_from_validator(state)
    assert next_node == "end"

test_route_validator_approved()

@test("Route from validator - rejected")
def test_route_validator_rejected():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"validation_results": {"approved": False}})
    next_node = route_from_validator(state)
    assert next_node == "debugger"

test_route_validator_rejected()

@test("Should retry - within limit")
def test_should_retry_within():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"retry_count": 1, "max_retries": 3})
    assert should_retry(state) is True

test_should_retry_within()

@test("Should retry - limit exceeded")
def test_should_retry_exceeded():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    state = update_state(state, {"retry_count": 3, "max_retries": 3})
    assert should_retry(state) is False

test_should_retry_exceeded()

@test("Needs human approval")
def test_needs_approval():
    task = {"id": "test", "description": "Test"}
    state = create_initial_state(task=task, project_id="test")
    assert needs_human_approval(state) is False

    state = update_state(state, {"requires_human_approval": True})
    assert needs_human_approval(state) is True

test_needs_approval()

print()
print("TEST SUITE 3: Agent Execution")
print("-" * 70)

from agents.coordinator import CoordinatorAgent
from agents.planner import PlannerAgent
from agents.architect import ArchitectAgent

@test("Coordinator agent execution")
async def test_coordinator_execute():
    agent = CoordinatorAgent(memory_service=None)
    task = {"id": "test", "description": "Test task"}
    state = create_initial_state(task=task, project_id="test")

    response = await agent.execute(state)

    assert response.success is True
    assert response.next_agent == AgentRole.PLANNER
    assert len(response.messages) > 0

test_coordinator_execute()

@test("Planner agent execution")
async def test_planner_execute():
    agent = PlannerAgent(memory_service=None)
    task = {"id": "test", "description": "Build app"}
    state = create_initial_state(task=task, project_id="test")

    response = await agent.execute(state)

    assert response.success is True
    assert "plan" in response.state_updates
    assert response.requires_human_approval is True

test_planner_execute()

@test("Architect agent execution")
async def test_architect_execute():
    agent = ArchitectAgent(memory_service=None)
    task = {"id": "test", "description": "Design system"}
    state = create_initial_state(task=task, project_id="test")
    state["plan"] = {"subtasks": []}

    response = await agent.execute(state)

    assert response.success is True
    assert "architecture" in response.state_updates

test_architect_execute()

print()
print("=" * 70)
print(f"RESULTS: {tests_passed}/{tests_run} tests passed, {tests_failed} failed")
if tests_failed == 0:
    print("SUCCESS: All unit tests passed!")
else:
    print(f"FAILURE: {tests_failed} test(s) failed")
print("=" * 70)

sys.exit(0 if tests_failed == 0 else 1)
