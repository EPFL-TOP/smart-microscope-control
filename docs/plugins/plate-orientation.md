# `plate-orientation`

Says which way the registered plate lies in its holder, and **renames its wells**. It
measures nothing and moves nothing.

A plate can be put in its holder either way round. Turned 180°, every well lands where
another one was (B3 where C4 was, on a 24-well): the positions are the same and only the
names change. The samples follow the names printed on the plate, so the run document must
name the wells the way the plate lies. The tools that fit a plate take the orientation as a
parameter ([`plate-registration`](plate-registration.md#which-way-the-plate-lies),
[`plate-walls`](plate-walls.md), [`plate-finding`](plate-finding.md#which-way-the-plate-lies)).
This tool is for when you find out **after** the fit.

## When to run it

When a well's printed label, or the cross on the plate page, says the plate lies the
other way from the registration: the stage is at the well the run document calls B3, and
the plate says C4.

```text
smc run plate-orientation -p <profile> --run-doc run.json --param orientation=turned-180
```

| Parameter | Default | Meaning |
|---|---|---|
| `orientation` | (required) | `not-turned` (A1 at the stage's -X +Y corner) or `turned-180` (A1 at the +X -Y corner). |

The tool is new: a microscope PC updated with `git pull` sees it only after
`pip install -e .` in the smc folder (a plugin is registered when smc is installed); until
then `smc run` answers `no plugin named 'plate-orientation'`.

It has no default: running it is a statement. It is **absolute**, not a switch: the
orientation the plate fit already has gives that fit back, number for number, so
running it twice with the same value changes nothing.

## What it does

It reads the latest plate fit (`plate-registration`, `plate-walls` or `plate-finding`,
never its own previous result) and writes it again under the new orientation:

- every well **position stays**: A1 moves to the old centre of the last well, and the
  rotation gains 180°;
- every well **name is turned**, row `r` to `R-1-r` and column `c` to `C-1-c`, in the
  result's references and in the walls `plate-walls` measured;
- the residual, the scale error, the objective and the method are the fit's;
- `assumed` keeps the fit's lines, whose well names are the fit's, and gains
  `turned 180° to turned-180 after the fit: every well renamed (row r to R-1-r, column c
  to C-1-c), nothing measured again; a well named in a line above is the fit's name`.

## What it never does

It never measures: it does not move the stage, take a frame or touch a device. A plate
that was **moved** in its holder needs a new registration, not this.

## After it

The tools that read the plate (`goto-well`, `scan-plan`, and what follows a plan) read
the renamed plate. Those run before it are **stale**: their well names changed, so
`smc run` prints `! scan-plan read plate from an earlier run; run it again`. Run them
again.

A plate fit run again later is the newer plate, and the renaming is set aside: declare
the orientation in that fit (`--param orientation=…`).

It refuses to run with no plate fit in the run document, or when the latest fit is not
usable (failed, stale, from another stand), with the run document's reason, as any tool
that reads the plate.
