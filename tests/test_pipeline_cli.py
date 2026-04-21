from scripts.video_gen.pipeline import build_arg_parser


def test_parser_requires_stage_or_refonly_or_scene():
    parser = build_arg_parser()
    args = parser.parse_args(["--scene", "day1_morning_commute", "--stage", "storyboard"])
    assert args.scene == "day1_morning_commute"
    assert args.stage == "storyboard"


def test_parser_accepts_regen_keyframe():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--scene", "day1_morning_commute",
        "--stage", "storyboard",
        "--regen-keyframe", "kf2",
    ])
    assert args.regen_keyframe == "kf2"


def test_parser_stage_all_is_valid():
    parser = build_arg_parser()
    args = parser.parse_args(["--scene", "day1_morning_commute", "--stage", "all"])
    assert args.stage == "all"
