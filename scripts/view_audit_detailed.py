#!/usr/bin/env python3
"""View detailed audit trail including full agent reasoning."""

import sys
from src.finvet.config.database import get_db_session
from src.finvet.audit.models import AuditEvent, AuditExecution

def view_detailed_audit(request_id=None):
    with get_db_session() as session:
        # Get latest request if none specified
        if not request_id:
            execution = session.query(AuditExecution).order_by(
                AuditExecution.execution_id.desc()
            ).first()
            request_id = execution.request_id if execution else None

        if not request_id:
            print("No requests found")
            return

        # Get execution summary
        execution = session.query(AuditExecution).filter(
            AuditExecution.request_id == request_id
        ).first()

        print("\n" + "="*80)
        print(f"DETAILED AUDIT TRAIL: {request_id}")
        print("="*80)
        print(f"Claim: {execution.claim_text}")
        print(f"Final Verdict: {execution.verdict} (confidence: {execution.confidence:.2f})")
        print(f"Execution Time: {execution.execution_time_ms}ms")
        print("="*80 + "\n")

        # Get events
        events = session.query(AuditEvent).filter(
            AuditEvent.request_id == request_id
        ).order_by(AuditEvent.timestamp).all()

        for i, event in enumerate(events, 1):
            data = event.data
            agent = event.agent or 'system'

            print(f"\n{'='*80}")
            print(f"{i}. [{event.event_type.upper()}] ({agent})")
            print(f"Time: {event.timestamp}")
            print("-"*80)

            if event.event_type == 'claim_parsed':
                print(f"Company: {data.get('company', 'N/A')}")
                print(f"Metric: {data.get('metric', 'N/A')}")
                print(f"Value Claimed: {data.get('parsed_claim', {}).get('value_claimed', 'N/A')}")
                print(f"Period: {data.get('parsed_claim', {}).get('period', 'N/A')}")
                print(f"Parse Confidence: {data.get('parse_confidence', 0):.2f}")

            elif event.event_type == 'agent_completed':
                print(f"✓ Verdict: {data.get('verdict', 'N/A')}")
                print(f"✓ Confidence: {data.get('confidence', 0):.2f}")
                print(f"✓ Sources: {data.get('sources_count', 0)}")
                print(f"✓ Execution Time: {data.get('execution_time_ms', 0)}ms")
                print("\nReasoning:")
                print(f"  {data.get('reasoning', 'N/A')}")
                print("\nEvidence Summary:")
                print(f"  {data.get('evidence_summary', 'N/A')[:500]}")
                
                # Show comparison if available
                if 'comparison' in data:
                    print("\nComparison:")
                    comp = data['comparison']
                    print(f"  Claimed: {comp.get('claimed_value', 'N/A')}")
                    print(f"  Actual: {comp.get('actual_value', 'N/A')}")
                    print(f"  Match: {comp.get('matches', 'N/A')}")

            elif event.event_type == 'consensus_reached':
                print(f"Final Verdict: {data.get('final_verdict', 'N/A')}")
                print(f"Final Confidence: {data.get('final_confidence', 0):.2f}")
                print(f"Primary Authority: {data.get('primary_authority', 'N/A')}")
                print("\nSummary:")
                print(f"  {data.get('summary', 'N/A')}")

if __name__ == '__main__':
    req_id = sys.argv[1] if len(sys.argv) > 1 else None
    view_detailed_audit(req_id)
