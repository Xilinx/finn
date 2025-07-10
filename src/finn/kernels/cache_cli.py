#!/usr/bin/env python3
"""
Command-line interface for managing the FINNKernel cache.

Provides utilities to inspect, clean, and manage the kernel cache.
"""

import argparse
import sys
from pathlib import Path

from .cache_manager import cache_manager
from . import gkr


def cmd_stats(args):
    """Display cache statistics."""
    stats = cache_manager.get_cache_stats()
    session_stats = cache_manager.get_session_stats()
    
    print("FINNKernel Cache Statistics")
    print("=" * 50)
    
    # Session statistics
    if session_stats['total_requests'] > 0:
        print("Session Statistics:")
        print(f"  Cache Hits: {session_stats['cache_hits']}")
        print(f"  Cache Misses: {session_stats['cache_misses']}")
        print(f"  Total Requests: {session_stats['total_requests']}")
        print(f"  Hit Rate: {session_stats['hit_rate_percent']:.1f}%")
        print()
        
        if args.details and (session_stats['hit_details'] or session_stats['miss_details']):
            print("Session Details:")
            if session_stats['hit_details']:
                print("  Cache Hits:")
                for kernel_name, kernel_class, op_type in session_stats['hit_details']:
                    print(f"    ✓ {kernel_name} ({kernel_class}, {op_type})")
            
            if session_stats['miss_details']:
                print("  Cache Misses:")
                for kernel_name, kernel_class, op_type in session_stats['miss_details']:
                    print(f"    ✗ {kernel_name} ({kernel_class}, {op_type})")
            print()
    
    # Persistent cache statistics
    print("Persistent Cache:")
    print(f"  Cache Directory: {stats['cache_dir']}")
    print(f"  Total Entries: {stats['total_entries']}")
    print(f"  Active Entries: {stats['active_entries']}")
    print(f"  Expired Entries: {stats['expired_entries']}")
    
    if stats['cache_size_bytes'] >= 0:
        print(f"  Cache Size: {stats['cache_size_mb']:.2f} MB ({stats['cache_size_bytes']} bytes)")
    else:
        print("  Cache Size: Unable to calculate")
    
    if stats['total_entries'] > 0 and args.entries:
        print("\nCache Entries:")
        for cache_hash, entry in cache_manager.metadata["entries"].items():
            status = "EXPIRED" if entry.get("expired", False) else "ACTIVE"
            kernel_class = entry.get('kernel_class', 'Unknown')
            
            # Determine kernel type from class name
            if 'HLS' in kernel_class:
                kernel_type = 'HLS'
            elif 'SIP' in kernel_class:
                kernel_type = 'SIP'
            elif 'RTL' in kernel_class:
                kernel_type = 'RTL'
            else:
                kernel_type = '???'
                
            print(f"  {cache_hash[:8]}... | {entry.get('op_type', 'Unknown')} | {kernel_class} ({kernel_type}) | {status}")


def cmd_clear(args):
    """Clear cache entries."""
    if args.all:
        count = gkr.clear_cache()
        print(f"Invalidated {count} cache entries")
    elif args.op_type:
        count = gkr.clear_cache(op_type=args.op_type)
        print(f"Invalidated {count} cache entries for operation type '{args.op_type}'")
    elif args.kernel_class:
        count = gkr.clear_cache(kernel_class=args.kernel_class)
        print(f"Invalidated {count} cache entries for kernel class '{args.kernel_class}'")
    else:
        print("Please specify --all, --op-type, or --kernel-class")
        return 1
    
    if args.cleanup:
        cleaned = gkr.cleanup_cache()
        print(f"Cleaned up {cleaned} expired entries from disk")
    
    return 0


def cmd_cleanup(args):
    """Clean up expired cache entries."""
    cleaned = gkr.cleanup_cache()
    print(f"Cleaned up {cleaned} expired cache entries from disk")
    
    if args.stats:
        print()
        cmd_stats(args)
    
    return 0


def cmd_check(args):
    """Check for shared file changes."""
    if args.op_type:
        changed_files = gkr.check_shared_files_changed(args.op_type)
        if changed_files:
            print(f"Shared files changed for operation type '{args.op_type}':")
            for file_path in changed_files:
                print(f"  {file_path}")
            
            if args.auto_invalidate:
                count = gkr.clear_cache(op_type=args.op_type)
                print(f"\nAuto-invalidated {count} cache entries due to shared file changes")
        else:
            print(f"No shared file changes detected for operation type '{args.op_type}'")
    else:
        # Check all operation types
        from .kernel_registry import KernelRegistry
        registry = KernelRegistry()
        
        all_changed = []
        for op_type in registry._mapping.keys():
            changed_files = gkr.check_shared_files_changed(op_type)
            if changed_files:
                all_changed.extend([(op_type, f) for f in changed_files])
        
        if all_changed:
            print("Shared file changes detected:")
            current_op = None
            for op_type, file_path in all_changed:
                if op_type != current_op:
                    print(f"\n{op_type}:")
                    current_op = op_type
                print(f"  {file_path}")
            
            if args.auto_invalidate:
                total_invalidated = 0
                for op_type in set(op for op, _ in all_changed):
                    count = gkr.clear_cache(op_type=op_type)
                    total_invalidated += count
                print(f"\nAuto-invalidated {total_invalidated} cache entries due to shared file changes")
        else:
            print("No shared file changes detected")
    
    return 0


def cmd_session(args):
    """Manage session statistics."""
    if args.reset:
        cache_manager.reset_session_stats()
        print("Session statistics reset")
        return 0
    
    # Default: show session stats
    session_stats = cache_manager.get_session_stats()
    
    print("FINNKernel Session Statistics")
    print("=" * 35)
    
    if session_stats['total_requests'] > 0:
        print(f"Cache Hits: {session_stats['cache_hits']}")
        print(f"Cache Misses: {session_stats['cache_misses']}")
        print(f"Total Requests: {session_stats['total_requests']}")
        print(f"Hit Rate: {session_stats['hit_rate_percent']:.1f}%")
        
        if args.details and (session_stats['hit_details'] or session_stats['miss_details']):
            print("\nDetails:")
            if session_stats['hit_details']:
                print("Cache Hits:")
                for kernel_name, kernel_class, op_type in session_stats['hit_details']:
                    print(f"  ✓ {kernel_name} ({kernel_class}, {op_type})")
            
            if session_stats['miss_details']:
                print("Cache Misses:")
                for kernel_name, kernel_class, op_type in session_stats['miss_details']:
                    print(f"  ✗ {kernel_name} ({kernel_class}, {op_type})")
    else:
        print("No cache requests in this session")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FINNKernel Cache Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                    # Show cache statistics
  %(prog)s stats --details          # Show cache stats with hit/miss details
  %(prog)s session                  # Show session statistics only
  %(prog)s session --details        # Show session stats with detailed list
  %(prog)s session --reset          # Reset session statistics
  %(prog)s clear --all             # Clear all cache entries
  %(prog)s clear --op-type FMPadding # Clear cache for FMPadding operations
  %(prog)s cleanup                 # Remove expired entries from disk
  %(prog)s check --op-type FMPadding # Check for shared file changes
  %(prog)s check --auto-invalidate  # Check all and auto-invalidate if needed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Display cache statistics')
    stats_parser.add_argument('--details', action='store_true', help='Show detailed session hit/miss information')
    stats_parser.add_argument('--entries', action='store_true', help='Show individual cache entries')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache entries')
    clear_group = clear_parser.add_mutually_exclusive_group(required=True)
    clear_group.add_argument('--all', action='store_true', help='Clear all cache entries')
    clear_group.add_argument('--op-type', help='Clear cache for specific operation type')
    clear_group.add_argument('--kernel-class', help='Clear cache for specific kernel class')
    clear_parser.add_argument('--cleanup', action='store_true', help='Also cleanup expired entries from disk')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up expired cache entries')
    cleanup_parser.add_argument('--stats', action='store_true', help='Show stats after cleanup')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check for shared file changes')
    check_parser.add_argument('--op-type', help='Check specific operation type (default: check all)')
    check_parser.add_argument('--auto-invalidate', action='store_true', help='Automatically invalidate affected cache entries')
    
    # Session command
    session_parser = subparsers.add_parser('session', help='Manage session statistics')
    session_parser.add_argument('--reset', action='store_true', help='Reset session statistics')
    session_parser.add_argument('--details', action='store_true', help='Show detailed hit/miss information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'stats':
        return cmd_stats(args)
    elif args.command == 'clear':
        return cmd_clear(args)
    elif args.command == 'cleanup':
        return cmd_cleanup(args)
    elif args.command == 'check':
        return cmd_check(args)
    elif args.command == 'session':
        return cmd_session(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())