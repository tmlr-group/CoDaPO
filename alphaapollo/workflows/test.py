from __future__ import annotations

from alphaapollo.workflows import api
from alphaapollo.workflows.common import parse_standard_args


def main() -> None:
    config, overrides = parse_standard_args(
        description="Run AlphaApollo test workflow.",
        default_config=api.DEFAULT_CONFIGS["test"],
    )
    api.test(config_path=config, extra_overrides=overrides)


if __name__ == "__main__":
    main()


