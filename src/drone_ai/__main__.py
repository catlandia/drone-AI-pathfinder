"""Main entry point for drone_ai package.

Run with:
    python -m drone_ai [command]

Commands:
    demo        - Run demo visualization
    map         - Show 3D movement map
    train       - Run curriculum learning
    plan        - Test path planning
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Drone AI Path Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  demo    Run demonstration with visualization
  map     Show interactive 3D movement map
  train   Run curriculum learning sequence
  plan    Test path planning algorithms

Examples:
  python -m drone_ai demo --task delivery_route --difficulty 0.7
  python -m drone_ai map --task delivery_route
  python -m drone_ai train --render --episodes 500
  python -m drone_ai plan
        """
    )

    parser.add_argument("command", nargs="?", default="demo",
                        choices=["demo", "map", "train", "plan"],
                        help="Command to run")
    parser.add_argument("--task", type=str, default="delivery_route",
                        choices=["hover", "delivery", "delivery_route"])
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)

    args, remaining = parser.parse_known_args()

    if args.command == "demo":
        from .demo import main as demo_main
        # Reconstruct argv for demo
        sys.argv = ["demo"] + remaining
        if args.task:
            sys.argv.extend(["--task", args.task])
        if args.difficulty:
            sys.argv.extend(["--difficulty", str(args.difficulty)])
        if args.render:
            sys.argv.append("--render")
        if args.episodes:
            sys.argv.extend(["--episodes", str(args.episodes)])
        if args.seed:
            sys.argv.extend(["--seed", str(args.seed)])
        demo_main()

    elif args.command == "train":
        from .learning_sequence import main as train_main
        sys.argv = ["train"] + remaining
        if args.render:
            sys.argv.append("--render")
        if args.episodes:
            sys.argv.extend(["--episodes", str(args.episodes)])
        if args.seed:
            sys.argv.extend(["--seed", str(args.seed)])
        train_main()

    elif args.command == "map":
        from .map_viewer import main as map_main
        sys.argv = ["map"] + remaining
        if args.task:
            sys.argv.extend(["--task", args.task])
        if args.difficulty:
            sys.argv.extend(["--difficulty", str(args.difficulty)])
        if args.seed:
            sys.argv.extend(["--seed", str(args.seed)])
        map_main()

    elif args.command == "plan":
        from .demo import test_path_planning
        test_path_planning()


if __name__ == "__main__":
    main()
