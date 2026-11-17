<div align="center">

# HITCH

**Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness**

[![Project page](https://img.shields.io/badge/Project-Page-4c8bf5)](https://hitch.axell.top)
[![arXiv](https://img.shields.io/badge/arXiv-2611.08039-b31b1b)](https://arxiv.org/abs/2611.08039)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HITCH-ffc107)](https://huggingface.co/Axellwppr/HITCH)
![Python](https://img.shields.io/badge/Python-3.10-3776ab)

</div>

HITCH is a low-level harness that lets a vision-language model operate a Unitree G1 humanoid through tool calls. The
model looks at the head-camera image, measures points in it and sends structured whole-body motion requests; the
harness plans them, hands them to a tracking controller and reports back. The same interface drives a MuJoCo
simulation and the real robot.

## Codex agent experiments in this fork

This fork preserves the HITCH control pipeline and adds two model entry points:
[Entry A](docs/DEEPSEEK.md) uses the original tool loop with a DeepSeek API;
[Entry B](docs/CODING.md) uses an installed Codex CLI to write and run Python programs
against the same persistent robot tools. See the [initial pilot results](docs/CODING_RESULTS.md).
Codex is an external dependency; API credentials and generated episode data are not part of this repository.

## Architecture

![The processes of the harness and the interfaces between them](docs/architecture.png)

The agent loop (or the `scripts/wbc` command line) talks to a **session**, which plans motion and records everything
under `data/`. The session reaches the robot side through two HTTP servers: the **link server**, which feeds the
tracking controller, and the **camera server**. Only what is below them differs between simulation (a MuJoCo bridge
and a simulated camera) and the real robot (the DDS bridge and the head camera).

## Install

Requirements: Linux (x86-64 or aarch64), Git and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Axellwppr/HITCH.git && cd HITCH
scripts/setup    # clones the tracking controller into third_party/ and creates .venv (Python 3.10)
```

## Quick start

Run the built-in task with a scripted model; no model server is needed. The command starts the simulation, opens a
window on it, runs the episode and stops the simulation again.

```bash
scripts/agent run tabletop_pick --backend fake --viewer
```

The window needs a display; on a headless machine (a server, an SSH session) leave out `--viewer`, and the episode
runs the same without it.

With our model ([HITCH-27B on Hugging Face](https://huggingface.co/Axellwppr/HITCH)), serve it under the name `hitch` at
`http://localhost:8000/v1` on a server with the OpenAI Responses API (for example vLLM; other servers:
[model backends](docs/USAGE.md#model-backends)), then:

```bash
scripts/agent run tabletop_pick --backend hitch --seed 0 --viewer
```

The command prints why the episode ended, the model's own claim (`success`) and the scene's success check (`check`).
The episode directory under `data/episodes/` has a readable `transcript.md` with the images.

## Simulation

`tabletop_pick` is a room with a table and one to three objects; the seed selects the objects, the layout and the
task (pick an object up, or put it in a bin, a bucket or on a mat). The robot has Dex3 hands. The simulation renders
with EGL; on a machine without a GPU set `MUJOCO_GL=osmesa`.

To drive the robot by hand instead of a model, open a session and send the steps yourself (`viewer=1` opens the
window; leave it out on a headless machine):

```bash
WBC_HAND=dex3 scripts/wbc connect 'sim://tabletop_pick?seed=0&viewer=1'   # prints the session address
export WBC_SESSION=http://127.0.0.1:PORT
scripts/wbc observe
scripts/wbc plan '{"walk": {"target": [0.3, 0]}}' && scripts/wbc execute && scripts/wbc wait
scripts/wbc close
```

All commands: [docs/USAGE.md](docs/USAGE.md). The request format: [docs/REFERENCE.md](docs/REFERENCE.md).

## Real robot

**The controller adds no safety limits of its own. An operator must supervise the robot and hold the remote's
emergency stop. The harness has no stop command: motion already queued plays to its end.**

Supported: a G1 with rubber hands and a RealSense D455 on a two-axis head. The setup, the connections and the
options are in [docs/REAL_ROBOT.md](docs/REAL_ROBOT.md); in short, on the robot computer:

```bash
scripts/setup && scripts/robot build              # once; also the udev rule (REAL_ROBOT.md)
sudo scripts/robot bridge IFACE                   # each in its own terminal, in this order;
scripts/robot controller                          # then start and A on the remote
scripts/robot link
scripts/robot vision
```

Then, with the two servers' ports forwarded to the machine that runs the harness, run a model on a task file of your
own:

```bash
scripts/agent run my_task.md --backend hitch --vision http://127.0.0.1:18764
```

## Documentation

| Document | Content |
| --- | --- |
| [docs/REAL_ROBOT.md](docs/REAL_ROBOT.md) | the real robot: hardware, start-up, depth repair, connecting, task files |
| [docs/USAGE.md](docs/USAGE.md) | commands and options, simulation targets, task and episode files, model backends, environment variables |
| [docs/REFERENCE.md](docs/REFERENCE.md) | the request a model sends, the execution rules, the refusal reasons and the feedback |
| [vision/README.md](vision/README.md) | the head-camera server |
| [scripts/lingbot/README.md](scripts/lingbot/README.md) | the depth-repair service |

## Citation

```bibtex
@misc{lu2026hitch,
  title         = {{HITCH}: Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness},
  author        = {Lu, Qingzhou and Liu, Chenxin and Wang, Xinming and Cao, Liu and Guo, Yanjiang and Xu, Mengdi and Chen, Jianyu},
  year          = {2026},
  eprint        = {2611.08039},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.08039}
}
```
