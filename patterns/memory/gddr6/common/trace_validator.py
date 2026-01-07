#!/usr/bin/env python3
"""
GDDR6 Memory Controller Trace Validator

Validates Chrome Trace JSON files against GDDR6 trace invariants.
See INVARIANTS.md for detailed invariant documentation.

Usage:
    python trace_validator.py <trace_file.json> [--verbose] [--json]

Exit codes:
    0 - All invariants pass
    1 - One or more invariants violated
    2 - Error reading/parsing trace file

SPDX-License-Identifier: MIT
Copyright (c) 2024-2026 Stillwater Supercomputing, Inc.
"""

import json
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
from pathlib import Path

# GDDR6-16000 timing parameters (cycles at 2.0 GHz CK)
TIMING = {
    'tRCDRD': 18,   # Row address to column address delay (read)
    'tRCDWR': 18,   # Row address to column address delay (write)
    'tRP': 18,      # Row precharge time
    'tRAS': 28,     # Row active time
    'tRC': 46,      # Row cycle time
    'tRL': 20,      # CAS read latency
    'tWL': 8,       # CAS write latency
    'tWR': 16,      # Write recovery
    'tRTP': 8,      # Read to precharge
    'tRRD_L': 4,    # ACT-to-ACT same bank group
    'tRRD_S': 4,    # ACT-to-ACT different bank group
    'tCCD_L': 3,    # CAS-to-CAS same bank group
    'tCCD_S': 2,    # CAS-to-CAS different bank group
    'tWTR_L': 6,    # Write-to-read turnaround same BG
    'tWTR_S': 4,    # Write-to-read turnaround diff BG
    'tRTW': 14,     # Read-to-write turnaround
    'tFAW': 16,     # Four activate window
    'tBurst': 4,    # BL16 burst cycles
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
    channel: Optional[int] = None
    events: List[Dict] = field(default_factory=list)
    fix_hint: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'invariant': self.invariant,
            'severity': self.severity.value,
            'message': self.message,
            'txn_id': self.txn_id,
            'bank': self.bank,
            'channel': self.channel,
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

        # Extract channel from bank_id if encoded
        bank_id = item.get('tid', 0)
        channel = bank_id // 16 if bank_id >= 16 else 0
        bank = bank_id % 16 if bank_id >= 16 else bank_id

        return cls(
            name=item.get('name', ''),
            txn_id=txn_id,
            cycle_issue=args.get('cycle_issue', 0),
            cycle_complete=args.get('cycle_complete', 0),
            bank=bank,
            channel=channel,
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
    def channel(self) -> Optional[int]:
        if self.events:
            return self.events[0].channel
        return None

    @property
    def cycle_start(self) -> int:
        return min(e.cycle_issue for e in self.events) if self.events else 0

    @property
    def cycle_end(self) -> int:
        return max(e.cycle_complete for e in self.events) if self.events else 0


class TraceValidator:
    """Validates GDDR6 memory controller traces against invariants."""

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
        self._check_inv106_tccd_constraint()
        self._check_inv107_tras_constraint()
        self._check_inv108_trc_constraint()

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
                        'cycle_complete': e.cycle_complete, 'bank': e.bank,
                        'channel': e.channel}
                       for e in txn.events],
                fix_hint="These commands should share txn_id with the READ/WRITE that triggered them"
            ))

    def _check_inv003_temporal_ordering(self):
        """INV-003: Commands must be temporally ordered correctly."""
        for txn_id, txn in self.transactions.items():
            if not txn.has_data_op:
                continue

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
                        channel=activate.channel,
                        fix_hint=f"Data operation must wait for ACTIVATE (tRCD={TIMING['tRCDRD']} cycles)"
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
                        channel=precharge.channel,
                        fix_hint="PRECHARGE must wait for data operation to complete"
                    ))

    def _check_inv004_unique_txn_ids(self):
        """INV-004: Each transaction ID should be used for exactly one logical request."""
        pass  # Covered by INV-001 multiple data operations check

    def _check_inv100_trcd_constraint(self):
        """INV-100: READ/WRITE must wait for ACTIVATE (tRCDRD/tRCDWR)."""
        for txn_id, txn in self.transactions.items():
            if not txn.has_data_op:
                continue

            activate = txn.activate
            data_op = txn.data_op

            if activate and data_op:
                gap = data_op.cycle_issue - activate.cycle_issue
                required = TIMING['tRCDRD'] if 'READ' in data_op.name else TIMING['tRCDWR']
                if gap < required:
                    self.violations.append(Violation(
                        invariant='INV-100',
                        severity=Severity.WARNING,
                        message=f"txn_id={txn_id}: tRCD violation - gap={gap} cycles, required={required}",
                        txn_id=txn_id,
                        bank=activate.bank,
                        channel=activate.channel,
                        fix_hint=f"Wait at least tRCD={required} cycles after ACTIVATE"
                    ))

    def _check_inv101_trp_constraint(self):
        """INV-101: ACTIVATE must wait for PRECHARGE (tRP)."""
        # Group events by (channel, bank)
        bank_events: Dict[tuple, List[Event]] = {}
        for event in self.events:
            key = (event.channel, event.bank)
            if key not in bank_events:
                bank_events[key] = []
            bank_events[key].append(event)

        for (channel, bank), events in bank_events.items():
            events.sort(key=lambda e: e.cycle_issue)

            last_precharge: Optional[Event] = None
            for event in events:
                if event.name == 'PRECHARGE':
                    last_precharge = event
                elif event.name == 'ACTIVATE' and last_precharge:
                    gap = event.cycle_issue - last_precharge.cycle_complete
                    if gap < 0:
                        self.violations.append(Violation(
                            invariant='INV-101',
                            severity=Severity.ERROR,
                            message=f"Ch{channel} Bank{bank}: ACTIVATE at cycle {event.cycle_issue} before "
                                   f"PRECHARGE completes at cycle {last_precharge.cycle_complete}",
                            bank=bank,
                            channel=channel,
                            txn_id=event.txn_id,
                            fix_hint=f"Wait for PRECHARGE to complete (tRP={TIMING['tRP']} cycles)"
                        ))

    def _check_inv102_trrd_constraint(self):
        """INV-102: Minimum time between ACTIVATE commands."""
        # Group activates by channel
        channel_activates: Dict[int, List[Event]] = {}
        for e in self.events:
            if e.name == 'ACTIVATE':
                if e.channel not in channel_activates:
                    channel_activates[e.channel] = []
                channel_activates[e.channel].append(e)

        for channel, activates in channel_activates.items():
            activates.sort(key=lambda e: e.cycle_issue)

            for i in range(1, len(activates)):
                prev = activates[i-1]
                curr = activates[i]
                gap = curr.cycle_issue - prev.cycle_issue

                # Determine if same bank group
                prev_bg = prev.bank // 4
                curr_bg = curr.bank // 4
                same_bg = prev_bg == curr_bg

                required = TIMING['tRRD_L'] if same_bg else TIMING['tRRD_S']

                if gap < required:
                    self.violations.append(Violation(
                        invariant='INV-102',
                        severity=Severity.WARNING,
                        message=f"Ch{channel} tRRD violation: ACT gap={gap} cycles between banks "
                               f"{prev.bank}->{curr.bank}, required={required} "
                               f"({'same BG' if same_bg else 'diff BG'})",
                        channel=channel,
                        fix_hint=f"tRRD_L={TIMING['tRRD_L']}, tRRD_S={TIMING['tRRD_S']}"
                    ))

    def _check_inv103_tfaw_constraint(self):
        """INV-103: Maximum 4 ACTIVATE commands in any tFAW window."""
        # Group activates by channel
        channel_activates: Dict[int, List[Event]] = {}
        for e in self.events:
            if e.name == 'ACTIVATE':
                if e.channel not in channel_activates:
                    channel_activates[e.channel] = []
                channel_activates[e.channel].append(e)

        tfaw = TIMING['tFAW']
        for channel, activates in channel_activates.items():
            activates.sort(key=lambda e: e.cycle_issue)

            for i, act in enumerate(activates):
                window_end = act.cycle_issue + tfaw
                count = sum(1 for a in activates if act.cycle_issue <= a.cycle_issue < window_end)

                if count > 4:
                    self.violations.append(Violation(
                        invariant='INV-103',
                        severity=Severity.WARNING,
                        message=f"Ch{channel} tFAW violation: {count} ACTIVATE commands in "
                               f"{tfaw}-cycle window starting at cycle {act.cycle_issue}",
                        channel=channel,
                        fix_hint=f"Maximum 4 ACTIVATE commands allowed in tFAW={tfaw} cycles"
                    ))
                    break

    def _check_inv106_tccd_constraint(self):
        """INV-106: Minimum time between CAS commands (tCCD)."""
        # Group CAS commands by channel
        channel_cas: Dict[int, List[Event]] = {}
        for e in self.events:
            if 'READ' in e.name or 'WRITE' in e.name:
                if e.channel not in channel_cas:
                    channel_cas[e.channel] = []
                channel_cas[e.channel].append(e)

        for channel, cas_events in channel_cas.items():
            cas_events.sort(key=lambda e: e.cycle_issue)

            for i in range(1, len(cas_events)):
                prev = cas_events[i-1]
                curr = cas_events[i]
                gap = curr.cycle_issue - prev.cycle_issue

                # Determine if same bank group
                prev_bg = prev.bank // 4
                curr_bg = curr.bank // 4
                same_bg = prev_bg == curr_bg

                required = TIMING['tCCD_L'] if same_bg else TIMING['tCCD_S']

                if gap < required:
                    self.violations.append(Violation(
                        invariant='INV-106',
                        severity=Severity.WARNING,
                        message=f"Ch{channel} tCCD violation: CAS gap={gap} cycles between banks "
                               f"{prev.bank}->{curr.bank}, required={required} "
                               f"({'same BG' if same_bg else 'diff BG'})",
                        channel=channel,
                        fix_hint=f"tCCD_L={TIMING['tCCD_L']}, tCCD_S={TIMING['tCCD_S']}"
                    ))

    def _check_inv107_tras_constraint(self):
        """INV-107: Minimum time a row must remain active (tRAS)."""
        for txn_id, txn in self.transactions.items():
            activate = txn.activate
            precharge = txn.precharge

            if activate and precharge:
                gap = precharge.cycle_issue - activate.cycle_issue
                if gap < TIMING['tRAS']:
                    self.violations.append(Violation(
                        invariant='INV-107',
                        severity=Severity.ERROR,
                        message=f"txn_id={txn_id}: tRAS violation - gap={gap} cycles, required={TIMING['tRAS']}",
                        txn_id=txn_id,
                        bank=activate.bank,
                        channel=activate.channel,
                        fix_hint=f"Row must remain active for at least tRAS={TIMING['tRAS']} cycles"
                    ))

    def _check_inv108_trc_constraint(self):
        """INV-108: Minimum row cycle time (tRC)."""
        # Group activates by (channel, bank)
        bank_activates: Dict[tuple, List[Event]] = {}
        for e in self.events:
            if e.name == 'ACTIVATE':
                key = (e.channel, e.bank)
                if key not in bank_activates:
                    bank_activates[key] = []
                bank_activates[key].append(e)

        for (channel, bank), activates in bank_activates.items():
            activates.sort(key=lambda e: e.cycle_issue)

            for i in range(1, len(activates)):
                prev = activates[i-1]
                curr = activates[i]
                gap = curr.cycle_issue - prev.cycle_issue

                if gap < TIMING['tRC']:
                    self.violations.append(Violation(
                        invariant='INV-108',
                        severity=Severity.ERROR,
                        message=f"Ch{channel} Bank{bank}: tRC violation - gap={gap} cycles between "
                               f"activates, required={TIMING['tRC']}",
                        bank=bank,
                        channel=channel,
                        fix_hint=f"Same-bank activates must be at least tRC={TIMING['tRC']} cycles apart"
                    ))

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
        print(f"GDDR6 Trace Validation Report")
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
                if v.channel is not None:
                    print(f"        channel: {v.channel}")
                if v.bank is not None:
                    print(f"        bank: {v.bank}")
                if v.events:
                    print(f"        events: {v.events[:3]}{'...' if len(v.events) > 3 else ''}")
                if v.fix_hint:
                    print(f"        FIX: {v.fix_hint}")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate GDDR6 memory controller traces against invariants'
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
