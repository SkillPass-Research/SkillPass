# Stilling the Cloudloom

The Cloudloom is an invented machine at the top of Harrowgate Spire that
spins rain clouds from river mist. Stilling it means bringing it to a full
stop so that nothing is spun until it is woken again. Use this document
whenever a task asks how the Cloudloom is stilled, which step comes next,
or what the operator reports.

## Before you start

The Cloudloom is always watched by a Loomwright, the keeper on duty in the
gallery. The operator never stills the loom alone if a Loomwright can be
raised. The drizzle gauge on the loom's flank reads in **mizzlets**; a
working loom reads between 8 and 14.

## The steps

Do these steps in this order. Each step has a label, and the operator
writes the label in the Spire log as each one is finished.

1. **CL-S1.** Ring the `hushbell` three times and wait for the Loomwright
   to answer from the gallery.
2. **CL-S2.** Close the vapour gate until its pointer rests on the notch
   marked `nimb-0`.
3. **CL-S3.** Lower the spindle crown, one turn at a time, until the
   drizzle gauge reads below 4 mizzlets.
4. **CL-S4.** Bleed the `thunderjar` by turning its brass tap anticlockwise
   until the jar falls silent.
5. **CL-S5.** Fold the cloud shuttle and lay it in its cradle.
6. **CL-S6.** Chalk the stilling mark, a circle with a bar through it, on
   the loom door.
7. **CL-S7.** Write the final report in the Spire log.

The final report is `LOOM-STILLED` when every step went as written.

## Exceptions

These override the steps above. Where two apply, apply both.

- **No answer to the hushbell.** If no Loomwright answers within twenty
  breaths at CL-S1, carry on from CL-S2 regardless, but the final report at
  CL-S7 is `LOOM-UNWATCHED` instead of LOOM-STILLED.
- **The vapour gate sticks.** If the pointer will not reach nimb-0 at
  CL-S2, stop at once. Do not lower the crown. The final report is
  `LOOM-JAMMED`, written straight away, and no other step is done.
- **A squallback.** If the drizzle gauge reads 9 mizzlets or more at any
  point during CL-S3, the gate has crept open. Go back to CL-S2, close the
  gate to nimb-0 again, and then resume CL-S3 from where you were.
- **The thunderjar hums.** If the thunderjar hums instead of falling silent
  during CL-S4, close the brass tap, ring the hushbell once, wait forty
  breaths, and begin CL-S4 again.
- **Never fold early.** CL-S5 is never done before CL-S4 is finished. A
  shuttle folded while the thunderjar still holds charge splits along its
  spine.

## Reporting

The Spire log holds one line per stilling: the labels of the steps done,
in order, then the final report. A stilling that ends in LOOM-JAMMED lists
only the steps actually done before the gate stuck.

## Answer format

When a task asks for a step, answer with its label, such as `CL-S3`. When
it asks for a report, answer with the report exactly as written above.
When it asks for several labels, separate them with a comma and one space.
