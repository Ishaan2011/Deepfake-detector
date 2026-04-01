# Third-Party Code and Assets

This repository contains original application code plus vendored or adapted third-party research code.

The root [LICENSE](LICENSE) applies to the original repository code written for this project. Third-party components keep their own copyright, license, attribution, and usage requirements.

## Included Third-Party Components

| Path | Component | Notes |
| --- | --- | --- |
| `MesoNet/` | MesoNet research implementation | Included for integration purposes. Pretrained weights are intentionally not redistributed with this repository. |
| `temp-d3/` | Temp-D3 / D3 research implementation | Included for integration purposes. Runtime may download upstream pretrained encoder assets on first use. |

## Redistribution Notes

- Do not assume you can relicense third-party directories under MIT unless you have verified the upstream terms yourself.
- Do not commit pretrained weights, datasets, cached model downloads, or downloaded source videos unless you have clear redistribution rights.
- If you publish a fork, keep upstream attribution intact and add any required notices for the components you ship.
