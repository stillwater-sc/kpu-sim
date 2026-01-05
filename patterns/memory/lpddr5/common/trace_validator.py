#!/usr/bin/env python3
"""
LPDDR5 Memory Controller Trace Validator

Validates Chrome Trace JSON files against LPDDR5 trace invariants.
See INVARIANTS.md for detailed invariant documentation.

Usage:
    python trace_validator.py <trace_file.json> [--verbose] [--json]

Exit codes:
    0 - All invariants pass
    1 - One or more invariants violated
    2 - Error reading/parsing trace file

SPDX-License-Identifier: MIT
Copyright (c) 2024-2025 Stillwater Supercomputing, Inc.
"""

import json
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
from pathlib import Path

# LPDDR5-6400 timing parameters (cycles at 3.2 GHz)
TIMING = {
    'tRCD': 14,    # Row address to column address delay
    'tRP': 14,     # Row precharge time
    'tCL': 14,     # CAS read latency
    'tRRD_L': 6,   # ACT-to-ACT same bank group
    'tRRD_S': 4,   # ACT-to-ACT different bank group
    'tFAW': 24,    # Four activate window
    'tRTW': 14,    # Read-to-write turnaround
    'tWTR_L': 10,  # Write-to-read turnaround
}


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Violation:
    invariant: str
    severity: Severity
    message: str
    txn_id: Optional[int] = None
    bank: Optional[int] = None
    events: List[Dict] = field(default_factory=list)
    fix_hint: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'invariant': self.invariant,
            'severity': self.severity.value,
            'message': self.message,
            'txn_id': self.txn_id,
            'bank': self.bank,
            'events': self.events,
            'fix_hint': self.fix_hint
        }


@dataclass
class Event:
    name: str
    txn_id: int
    cycle_issue: int
    cycle_complete: int
    bank: int
    channel: int = 0
    raw: Dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, item: Dict) -> Optional['Event']:
        """Parse a trace event from JSON."""
        if item.get('ph') == 'M':  # Skip metadata
            return None

        args = item.get('args', {})
        txn_id = args.get('txn_id')
        if txn_id is None:
            return None

        return cls(
            name=item.get('name', ''),
            txn_id=txn_id,
            cycle_issue=args.get('cycle_issue', 0),
            cycle_complete=args.get('cycle_complete', 0),
            bank=item.get('tid', 0),
            channel=0,  # Extract from pid if needed
            raw=item
        )


@dataclass
class Transaction:
    txn_id: int
    events: List[Event] = field(default_factory=list)

    @property
    def activate(self) -> Optional[Event]:
        return next((e for e in self.events if e.name == 'ACTIVATE'), None)

    @property
    def precharge(self) -> Optional[Event]:
        return next((e for e in self.events if e.name == 'PRECHARGE'), None)

    @property
    def data_op(self) -> Optional[Event]:
        return next((e for e in self.events
                    if 'READ' in e.name or 'WRITE' in e.name), None)

    @property
    def has_read(self) -> bool:
        return any('READ' in e.name for e in self.events)

    @property
    def has_write(self) -> bool:
        return any('WRITE' in e.name for e in self.events)

    @property
    def has_data_op(self) -> bool:
        return self.has_read or self.has_write

    @property
    def request_type(self) -> Optional[str]:
        if self.has_read:
            return 'READ'
        elif self.has_write:
            return 'WRITE'
        return None

    @property
    def bank(self) -> Optional[int]:
        if self.events:
            return self.events[0].bank
        return None

    @property
    def cycle_start(self) -> int:
        return min(e.cycle_issue for e in self.events) if self.events else 0

    @property
    def cycle_end(self) -> int:
        return max(e.cycle_complete for e in self.events) if self.events else 0


class TraceValidator:
    """Validates LPDDR5 memory controller traces against invariants."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.violations: List[Violation] = []
        self.events: List[Event] = []
        self.transactions: Dict[int, Transaction] = {}

    def load_trace(self, filepath: str) -> bool:
        """Load and parse a trace file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Parse events
            self.events = []
            for item in data:
                event = Event.from_json(item)
                if event:
                    self.events.append(event)

            # Sort by cycle_issue
            self.events.sort(key=lambda e: e.cycle_issue)

            # Group by txn_id
            self.transactions = {}
            for event in self.events:
                if event.txn_id not in self.transactions:
                    self.transactions[event.txn_id] = Transaction(event.txn_id)
                self.transactions[event.txn_id].events.append(event)

            if self.verbose:
                print(f"Loaded {len(self.events)} events, {len(self.transactions)} transactions")

            return True

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {filepath}: {e}", file=sys.stderr)
            return False
        except FileNotFoundError:
            print(f"ERROR: File not found: {filepath}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"ERROR: Failed to load trace: {e}", file=sys.stderr)
            return False

    def validate(self) -> bool:
        """Run all invariant checks. Returns True if all pass."""
        self.violations = []

        # Structure invariants
        self._check_inv001_valid_txn_id_semantics()
        self._check_inv002_command_ownership()
        self._check_inv003_temporal_ordering()
        self._check_inv004_unique_txn_ids()

        # Timing invariants
        self._check_inv100_trcd_constraint()
        self._check_inv101_trp_constraint()
        self._check_inv102_trrd_constraint()
        self._check_inv103_tfaw_constraint()

        return len([v for v in self.violations if v.severity == Severity.ERROR]) == 0

    def _check_inv001_valid_txn_id_semantics(self):
        """INV-001: Every txn_id must have exactly ONE data operation."""
        for txn_id, txn in self.transactions.items():
            # Count data operations
            read_count = sum(1 for e in txn.events if 'READ' in e.name)
            write_count = sum(1 for e in txn.events if 'WRITE' in e.name)
            data_op_count = read_count + write_count

            if data_op_count == 0:
                # No data operation - this txn_id is invalid
                event_names = [e.name for e in txn.events]
                self.violations.append(Violation(
                    invariant='INV-001',
                    severity=Severity.ERROR,
                    message=f"txn_id={txn_id} has no data operation (only {', '.join(set(event_names))} events)",
                    txn_id=txn_id,
                    events=[{'name': e.name, 'cycle_issue': e.cycle_issue,
                            'cycle_complete': e.cycle_complete} for e in txn.events],
                    fix_hint="ACTIVATE/PRECHARGE should have same txn_id as the READ/WRITE that triggered them"
                ))
            elif data_op_count > 1:
                # Multiple data operations with same txn_id
                self.violations.append(Violation(
                    invariant='INV-001',
                    severity=Severity.ERROR,
                    message=f"txn_id={txn_id} has {data_op_count} data operations (should be exactly 1)",
                    txn_id=txn_id,
                    events=[{'name': e.name, 'cycle_issue': e.cycle_issue}
                           for e in txn.events if 'READ' in e.name or 'WRITE' in e.name],
                    fix_hint="Each user request should have a unique txn_id"
                ))

            if read_count > 0 and write_count > 0:
                # Both READ and WRITE with same txn_id
                self.violations.append(Violation(
                    invariant='INV-001',
                    severity=Severity.ERROR,
                    message=f"txn_id={txn_id} has both READ and WRITE operations",
                    txn_id=txn_id,
                    fix_hint="A transaction must be either READ or WRITE, not both"
                ))

    def _check_inv002_command_ownership(self):
        """INV-002: ACTIVATE/PRECHARGE must belong to valid transactions."""
        # Find txn_ids that only have page management commands
        orphan_txn_ids = [
            txn_id for txn_id, txn in self.transactions.items()
            if not txn.has_data_op and (txn.activate or txn.precharge)
        ]

        for txn_id in orphan_txn_ids:
            txn = self.transactions[txn_id]
            cmd_names = [e.name for e in txn.events]
            self.violations.append(Violation(
                invariant='INV-002',
                severity=Severity.ERROR,
                message=f"txn_id={txn_id} has orphaned page management commands: {', '.join(cmd_names)}",
                txn_id=txn_id,
                events=[{'name': e.name, 'cycle_issue': e.cycle_issue,
                        'cycle_complete': e.cycle_complete, 'bank': e.bank}
                       for e in txn.events],
                fix_hint="These commands should share txn_id with the READ/WRITE that triggered them"
            ))

    def _check_inv003_temporal_ordering(self):
        """INV-003: Commands must be temporally ordered correctly."""
        for txn_id, txn in self.transactions.items():
            if not txn.has_data_op:
                continue  # Skip invalid transactions (caught by INV-001)

            data_op = txn.data_op
            activate = txn.activate
            precharge = txn.precharge

            # ACTIVATE must complete before data operation starts
            if activate and data_op:
                if data_op.cycle_issue < activate.cycle_complete:
                    self.violations.append(Violation(
                        invariant='INV-003',
                        severity=Severity.ERROR,
                        message=f"txn_id={txn_id}: {data_op.name} starts at cycle {data_op.cycle_issue} "
                               f"before ACTIVATE completes at cycle {activate.cycle_complete}",
                        txn_id=txn_id,
                        bank=activate.bank,
                        fix_hint=f"Data operation must wait for ACTIVATE (tRCD={TIMING['tRCD']} cycles)"
                    ))

            # PRECHARGE must start after data operation completes
            if precharge and data_op:
                if precharge.cycle_issue < data_op.cycle_complete:
                    self.violations.append(Violation(
                        invariant='INV-003',
                        severity=Severity.ERROR,
                        message=f"txn_id={txn_id}: PRECHARGE starts at cycle {precharge.cycle_issue} "
                               f"before {data_op.name} completes at cycle {data_op.cycle_complete}",
                        txn_id=txn_id,
                        bank=precharge.bank,
                        fix_hint="PRECHARGE must wait for data operation to complete"
                    ))

    def _check_inv004_unique_txn_ids(self):
        """INV-004: Each transaction ID should be used for exactly one logical request."""
        # This is partially covered by INV-001, but we add a check for ID reuse patterns
        pass  # Covered by INV-001 multiple data operations check

    def _check_inv100_trcd_constraint(self):
        """INV-100: READ/WRITE must wait for ACTIVATE (tRCD)."""
        for txn_id, txn in self.transactions.items():
            if not txn.has_data_op:
                continue

            activate = txn.activate
            data_op = txn.data_op

            if activate and data_op:
                gap = data_op.cycle_issue - activate.cycle_issue
                if gap < TIMING['tRCD']:
                    self.violations.append(Violation(
                        invariant='INV-100',
                        severity=Severity.WARNING,
                        message=f"txn_id={txn_id}: tRCD violation - gap={gap} cycles, required={TIMING['tRCD']}",
                        txn_id=txn_id,
                        bank=activate.bank,
                        fix_hint=f"Wait at least tRCD={TIMING['tRCD']} cycles after ACTIVATE"
                    ))

    def _check_inv101_trp_constraint(self):
        """INV-101: ACTIVATE must wait for PRECHARGE (tRP)."""
        # Group events by bank
        bank_events: Dict[int, List[Event]] = {}
        for event in self.events:
            if event.bank not in bank_events:
                bank_events[event.bank] = []
            bank_events[event.bank].append(event)

        for bank, events in bank_events.items():
            events.sort(key=lambda e: e.cycle_issue)

            last_precharge: Optional[Event] = None
            for event in events:
                if event.name == 'PRECHARGE':
                    last_precharge = event
                elif event.name == 'ACTIVATE' and last_precharge:
                    gap = event.cycle_issue - last_precharge.cycle_complete
                    if gap < 0:  # ACTIVATE before PRECHARGE completes
                        self.violations.append(Violation(
                            invariant='INV-101',
                            severity=Severity.ERROR,
                            message=f"Bank {bank}: ACTIVATE at cycle {event.cycle_issue} before "
                                   f"PRECHARGE completes at cycle {last_precharge.cycle_complete}",
                            bank=bank,
                            txn_id=event.txn_id,
                            fix_hint=f"Wait for PRECHARGE to complete (tRP={TIMING['tRP']} cycles)"
                        ))

    def _check_inv102_trrd_constraint(self):
        """INV-102: Minimum time between ACTIVATE commands."""
        activates = [e for e in self.events if e.name == 'ACTIVATE']
        activates.sort(key=lambda e: e.cycle_issue)

        for i in range(1, len(activates)):
            prev = activates[i-1]
            curr = activates[i]
            gap = curr.cycle_issue - prev.cycle_issue

            # Determine if same bank group (simplified: banks 0-3 = BG0, 4-7 = BG1, etc.)
            prev_bg = prev.bank // 4
            curr_bg = curr.bank // 4
            same_bg = prev_bg == curr_bg

            required = TIMING['tRRD_L'] if same_bg else TIMING['tRRD_S']

            if gap < required:
                self.violations.append(Violation(
                    invariant='INV-102',
                    severity=Severity.WARNING,
                    message=f"tRRD violation: ACT gap={gap} cycles between banks {prev.bank}->{curr.bank}, "
                           f"required={required} ({'same BG' if same_bg else 'diff BG'})",
                    fix_hint=f"tRRD_L={TIMING['tRRD_L']}, tRRD_S={TIMING['tRRD_S']}"
                ))

    def _check_inv103_tfaw_constraint(self):
        """INV-103: Maximum 4 ACTIVATE commands in any tFAW window."""
        activates = [e for e in self.events if e.name == 'ACTIVATE']
        activates.sort(key=lambda e: e.cycle_issue)

        tfaw = TIMING['tFAW']
        for i, act in enumerate(activates):
            window_end = act.cycle_issue + tfaw
            count = sum(1 for a in activates if act.cycle_issue <= a.cycle_issue < window_end)

            if count > 4:
                self.violations.append(Violation(
                    invariant='INV-103',
                    severity=Severity.WARNING,
                    message=f"tFAW violation: {count} ACTIVATE commands in {tfaw}-cycle window starting at cycle {act.cycle_issue}",
                    fix_hint=f"Maximum 4 ACTIVATE commands allowed in tFAW={tfaw} cycles"
                ))
                break  # Only report first violation

    def get_report(self) -> Dict:
        """Generate validation report."""
        errors = [v for v in self.violations if v.severity == Severity.ERROR]
        warnings = [v for v in self.violations if v.severity == Severity.WARNING]

        valid_txns = [t for t in self.transactions.values() if t.has_data_op]
        invalid_txns = [t for t in self.transactions.values() if not t.has_data_op and t.events]

        return {
            'status': 'PASSED' if len(errors) == 0 else 'FAILED',
            'violations': [v.to_dict() for v in self.violations],
            'summary': {
                'total_events': len(self.events),
                'total_transactions': len(self.transactions),
                'valid_transactions': len(valid_txns),
                'invalid_transactions': len(invalid_txns),
                'errors': len(errors),
                'warnings': len(warnings)
            }
        }

    def print_report(self, json_output: bool = False):
        """Print validation report."""
        report = self.get_report()

        if json_output:
            print(json.dumps(report, indent=2))
            return

        # Human-readable output
        status = report['status']
        summary = report['summary']

        print(f"\n{'='*60}")
        print(f"LPDDR5 Trace Validation Report")
        print(f"{'='*60}")
        print(f"Status: {status}")
        print(f"Events: {summary['total_events']}")
        print(f"Transactions: {summary['valid_transactions']} valid, {summary['invalid_transactions']} invalid")
        print(f"Errors: {summary['errors']}, Warnings: {summary['warnings']}")

        if self.violations:
            print(f"\n{'='*60}")
            print("Violations:")
            print(f"{'='*60}")

            for v in self.violations:
                icon = "ERROR" if v.severity == Severity.ERROR else "WARN "
                print(f"\n[{icon}] {v.invariant}: {v.message}")
                if v.txn_id is not None:
                    print(f"        txn_id: {v.txn_id}")
                if v.bank is not None:
                    print(f"        bank: {v.bank}")
                if v.events:
                    print(f"        events: {v.events[:3]}{'...' if len(v.events) > 3 else ''}")
                if v.fix_hint:
                    print(f"        FIX: {v.fix_hint}")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate LPDDR5 memory controller traces against invariants'
    )
    parser.add_argument('trace_file', help='Path to trace JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    validator = TraceValidator(verbose=args.verbose)

    if not validator.load_trace(args.trace_file):
        sys.exit(2)

    passed = validator.validate()
    validator.print_report(json_output=args.json)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
