# Thistlewane Ferry Relay

The Thistlewane Ferry Relay is the internal service that books space on a
ferry and loads crates onto it. Every client talks to it through four calls,
and the relay rejects any session that makes them out of order. Use this
document whenever a task asks how to ship a crate through the relay, or what
one of its error codes means.

## Call order

A session makes these calls in exactly this order, once each per crate:

1. `twf_greet` opens the session. It takes one parameter, `vessel_tag`, the
   tag of the ferry the client wants. It returns a `ferry_sigil`, a short
   string the client must keep for the last call.
2. `twf_reserve_berth` reserves space. It takes `berth_quota`, the number of
   berths wanted, and returns a `tidemark`. A tidemark is valid for 90
   seconds.
3. `twf_load_crate` puts one crate in the reserved berth. It takes
   `crate_glyph`, the crate's label, and `heft_in_grains`, its weight. It
   also takes the `tidemark` from the previous call.
4. `twf_cast_off` confirms the shipment. It takes the `ferry_sigil` returned
   by `twf_greet`, unchanged.

Never skip a call, and never call `twf_cast_off` before `twf_load_crate` has
succeeded.

## Limits

- A single crate may weigh at most 7200 grains. Split anything heavier into
  several crates, each with its own `crate_glyph`.
- A `berth_quota` above 12 is refused; open a second session instead.

## Error codes

Every error has a numeric code and a label. Report both when asked.

| Code | Label | Cause | What the client does next |
|---|---|---|---|
| TWF-117 | SILTED_BERTH | `twf_reserve_berth` was called before `twf_greet` | Start again from `twf_greet` |
| TWF-240 | CRATE_OVERHEFT | `heft_in_grains` is above 7200 | Split the crate and call `twf_load_crate` again |
| TWF-352 | SIGIL_MISMATCH | The `ferry_sigil` sent to `twf_cast_off` differs from the one issued | Start again from `twf_greet`; the old sigil is void |
| TWF-409 | TIDE_EXPIRED | The `tidemark` is older than 90 seconds | Call `twf_reserve_berth` again for a fresh tidemark; do not greet again |
| TWF-503 | FERRY_BECALMED | The ferry is not accepting departures | Wait 11 seconds and retry `twf_cast_off`, at most 3 times |

## Retries

Only TWF-503 is retried as is. Every other error changes where the client
restarts, as the table says. After a third TWF-503 in a row, stop and report
FERRY_BECALMED to the caller rather than retrying again.

## Answer format

When a task asks for a code and its label together, write the code, one
space, then the label, for example `TWF-117 SILTED_BERTH`. When it asks for a
sequence of calls, write the call names separated by a comma and one space.
