#!/usr/bin/env python3
"""
Test handoff extraction logic independently
"""

# Sample workflow data (simulating API response)
workflows = [
    {
        "workflow_id": "test1",
        "agents_executed": ["analytics_evaluation", "marketing_strategy"],
        "state_transitions": [],
    },
    {
        "workflow_id": "test2",
        "agents_executed": ["customer_support", "feedback_learning"],
        "state_transitions": [],
    },
    {
        "workflow_id": "test3",
        "agents_executed": ["marketing_strategy"],
        "state_transitions": [],
    },
]

# Simulate the handoff extraction logic
handoff_counts = {}
workflows_with_multiple_agents = 0

for workflow in workflows:
    handoff_found = False
    agents_exec = workflow.get("agents_executed", [])

    # Debug: Count workflows with multiple agents
    if len(agents_exec) > 1:
        workflows_with_multiple_agents += 1
        print(
            f"✓ Workflow {workflow['workflow_id']} has {len(agents_exec)} agents: {agents_exec}"
        )

    # Method 1: state_transitions (empty in our case)
    if "state_transitions" in workflow and workflow["state_transitions"]:
        print(f"  Using state_transitions for {workflow['workflow_id']}")
        # ...would process here...
        handoff_found = True

    # Method 2: Extract from agents_executed when multiple agents were involved
    if not handoff_found and (
        "agents_executed" in workflow and len(workflow.get("agents_executed", [])) > 1
    ):
        agents = workflow["agents_executed"]
        print(f"  Processing agents_executed for {workflow['workflow_id']}: {agents}")
        for i in range(len(agents) - 1):
            from_agent = agents[i].replace("_", " ").title()
            to_agent = agents[i + 1].replace("_", " ").title()

            print(f"    Handoff: {from_agent} → {to_agent}")

            if from_agent != to_agent:
                key = f"{from_agent}-{to_agent}"
                handoff_counts[key] = handoff_counts.get(key, 0) + 1
                handoff_found = True

print(f"\n{'='*60}")
print(f"Results:")
print(f"  Workflows with multiple agents: {workflows_with_multiple_agents}")
print(f"  Handoff patterns found: {len(handoff_counts)}")
print(f"\nHandoff counts:")
for handoff, count in handoff_counts.items():
    print(f"  {handoff}: {count}")

if not handoff_counts:
    print("\n❌ PROBLEM: No handoffs extracted!")
else:
    print(f"\n✅ SUCCESS: Extracted {len(handoff_counts)} handoff patterns")
