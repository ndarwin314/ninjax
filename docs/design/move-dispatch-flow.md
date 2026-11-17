# Move dispatch control flow

These diagrams describe the current **singles implementation**, not a complete
Scarlet/Violet rules specification. Read them in order: the first gives the
whole path; the others expand its major components. The main entry is
[`Battle.step`](../../ninjax/battle/environment.py); move execution is split between
[`battle.py`](../../ninjax/battle/environment.py) and the
[`moves`](../../ninjax/moves/) package. Selection and called moves live in
[`moves.execution`](../../ninjax/moves/execution.py); accuracy, immunity and
hit branch selection live in [`moves.routing`](../../ninjax/moves/routing.py).
See [move mechanics ownership](moves-package.md) for the remaining modules.

Arrows describe logical runtime paths. Branches and loops are primarily
`jax.lax.cond`, `switch`, `fori_loop`, and `scan`: they are traced JAX control
flow, not Python branching on battle values. State is an immutable PyTree;
random draws receive and return explicit keys. An event or hit-result record is
observational bookkeeping, while the battle state drives mechanics.

## 1. High-level path

```mermaid
flowchart TD
    A["Battle.step: two player actions"] --> B["Validate phase, encoding and legal masks"]
    B -->|"Invalid or terminal"| Z["Keep transition state; build timestep"]
    B -->|"Valid team preview"| P["Select roster and leads; activate entry abilities"]
    B -->|"Valid turn or replacement"| C["_advance: order or resume queued actions"]
    C --> D["For each eligible queued actor: step_action"]
    D -->|"Attack or charge release"| E["step_move: prepare and check interruptions"]
    D -->|"Switch / recharge / no-op"| F["Switch, spend recharge, or leave state unchanged"]
    E -->|"Not interrupted"| G["move_used: PP, special setup and dispatch"]
    E -->|"Interrupted"| H["move_interrupted; record move result"]
    G --> I["Accuracy and hit routing"]
    I --> J["Damage / status / ability immunity branch"]
    J --> K["Per-hit and per-move effects; resolve faints"]
    H --> L["Scheduler updates reveals and pivot request"]
    K --> L
    F --> L
    L -->|"Pivot with legal bench"| M["Pause for replacement; preserve queue and cursor"]
    M --> B
    L -->|"Another queued action"| D
    L -->|"Queue complete"| N["step_field: residual effects and turn advance"]
    N --> O["Request faint replacements or next turn"]
    O --> Z
    P --> Z
```

The initial preview path requires a structured `BattleAction`. A rejected active
player action leaves the transition state unchanged and reports `invalid_action`.
Terminal status, reward, observation, legal masks and timestep are constructed by
`step` after normal advancement. A timeout is a draw under the current policy.

## 2. Scheduling, pivots and replacements

```mermaid
flowchart TD
    A["_advance"] --> B{"Current phase?"}
    B -->|"TURN"| C["select: split ordering key; compute effective priorities"]
    C --> D["action_order: priority, Speed / Trick Room, random tie"]
    D --> E["Save pending_actions, original actor slots, order, cursor=0; field_pending=true"]
    B -->|"REPLACEMENT"| F["replace_active: switch each acting player with one loop"]
    F --> G["Clear acting-player mask; resume saved cursor"]
    E --> H["Up to two resolve iterations"]
    G --> H
    H --> I{"cursor < 2 and no pending replacement?"}
    I -->|"No"| Q["Stop action loop"]
    I -->|"Yes"| J["Set pending opponent move, type and original slot context"]
    J --> K{"Queued actor alive and still in original slot?"}
    K -->|"Yes"| L["step_action"]
    K -->|"No"| M["Skip displaced or fainted actor"]
    L --> N["refresh_paradox; capture conditions; reveal spent/called moves"]
    M --> N
    N --> O{"Switch requested, actor alive, selected living bench exists?"}
    O -->|"Yes"| P["Advance cursor; set REPLACEMENT and acting mask; suspend turn"]
    O -->|"No"| R["Advance cursor; continue loop"]
    R --> H
    P --> S["Return decision point"]
    Q --> T{"Queue complete, no replacement waiting, field_pending?"}
    T -->|"Yes"| U["step_field once; turn += 1; field_pending=false"]
    T -->|"No"| V["Do not repeat field effects"]
    U --> W["Find fainted actives with living selected replacements"]
    V --> W
    W --> S
```

A pivot pauses immediately, then `replace_active` resumes the saved queue. A
faint normally waits until queued actions and field effects finish. `field_pending`
prevents a replacement following a completed turn from running residual effects a
second time. Entry hazards and entry abilities are inside `swap_out`; an entry
faint can lead to another replacement decision. The scheduler also cancels a
queued action when its original active slot has been displaced.

## 3. Action and pre-move handling

```mermaid
flowchart TD
    A["step_action: set event source/cause; convert legacy action if needed"] --> B{"Transformation is NONE?"}
    B -->|"No"| R["No action: unsupported transformation"]
    B -->|"Yes"| C["_ordinary_action"]
    C --> D{"Action kind"}
    D -->|"Switch"| E["step_switch -> swap_out"]
    D -->|"Continued and recharging"| F["_spend_recharge: clear recharge and record skipped"]
    D -->|"No-op or invalid continuation"| R
    D -->|"Attack or charge release"| G["step_move"]
    G --> H{"Preparation kind"}
    H -->|"Struggle index 4"| I["_prepare_struggle_action: count attempt; no slot PP"]
    H -->|"Charge release"| J["_prepare_charge_release: find stored move; clear charge; check Disable"]
    H -->|"Selected move"| K["_prepare_selected_action: resolve Encore/restrictions; count attempt; retain queued priority"]
    I --> L["before_move"]
    J --> L
    K --> L
    L --> M["Decrement sleep; check freeze, flinch and restrictions"]
    M --> N["If eligible, decrement confusion and possibly self-hit"]
    N --> O["Roll paralysis; combine interruption flags"]
    O -->|"Interrupted"| P["move_interrupted"]
    O -->|"Can act"| Q["move_used"]
    P --> S["Classify move_result and resolve_faints"]
    Q --> S
    S --> T["step_action resolves pending faints again"]
    E --> T
    F --> T
    R --> T
```

Recharge bypasses `before_move`. A charge release does not pay PP again.
Struggle uses a synthetic move and has its own recoil path. In the public action
encoder, non-`NONE` transformations are rejected; the internal `is_tera`
parameter is not evidence of a complete transformation lifecycle.

## 4. Move preparation and special dispatch

```mermaid
flowchart TD
    A["move_used"] --> B{"Use kind"}
    B -->|"Selected"| C["_prepare_move_use: clear Destiny Bond as applicable; thaw; history and event; Choice lock; type change; spend PP"]
    B -->|"Charge release"| D["Announce stored move; skip selection bookkeeping and PP"]
    B -->|"Struggle"| E["Set event_move=-1; announce; clear move history"]
    C --> F["_execute_selected_or_called_move"]
    D --> F
    E --> F
    F --> G{"Sleep Talk selection?"}
    G -->|"Yes"| H{"Asleep and eligible called move?"}
    H -->|"No"| X["Record failure; no called dispatch"]
    H -->|"Yes"| I["Sample eligible slot; set called-move and sleep context; apply type change"]
    G -->|"No"| J["dispatch_selected_move"]
    I --> J
    J --> K{"Initial selected move?"}
    K -->|"Continued or Struggle"| O["Ready for shared hit pipeline"]
    K -->|"Yes"| L{"Fake Out / Sucker Punch failure?"}
    L -->|"Yes"| M["Record FAILED after PP payment"]
    L -->|"No"| N{"Explosion or Electro Shot?"}
    N -->|"Explosion"| N1["Damp blocks; otherwise save HP and self-faint"]
    N -->|"Electro Shot"| N2["Raise SpA; rain attacks now, otherwise store charge"]
    N -->|"Other"| O
    N1 --> O
    N2 --> O
    O -->|"Ready and implemented, or Struggle"| P["_apply_move_effects"]
    O -->|"Unsupported"| Q["move_misses: record MISS"]
    O -->|"Not ready"| U["Skip hit pipeline"]
    P --> U
    Q --> U
    M --> U
    U --> V["Clear Explosion saved HP if used"]
    V --> W["finish_attempt for ordinary initial dispatch: crash damage and update_items"]
    W --> Y["Restore Sleep Talk context when applicable"]
    X --> Y
    Y --> Z["Apply Struggle recoil if applicable; return"]
```

For continued moves and Struggle, `dispatch_selected_move` skips its ordinary
`finish_attempt` branch. Sleep Talk spends its own PP but calls its selected
move without spending that move's PP; it preserves and restores calling context.
Unsupported catalogue moves are generally unavailable through the legal mask;
if reached in dispatch, the code records a miss rather than executing effects.

## 5. Accuracy and hit routing

```mermaid
flowchart TD
    A["_apply_move_effects"] --> B{"Struggle?"}
    B -->|"Yes"| G["_move_hits"]
    B -->|"No"| C["_prepare_accuracy: split key; effective_move; move_accuracy"]
    C --> D{"Worry Seed forbidden identity?"}
    D -->|"Yes"| E["Record IMMUNE; stop"]
    D -->|"No"| F{"Accuracy succeeds or represented protection applies?"}
    F -->|"No"| H["move_misses: record MISS and event"]
    F -->|"Yes"| G
    G --> I{"Struggle?"}
    I -->|"Yes"| J["_prepare_struggle_hit: typeless 50-power physical move; branch 0"]
    I -->|"No"| K["_prepare_normal_hit: effective move; queued priority override"]
    K --> L["Choose damage, status, absorption boost/heal, Flash Fire, or immunity"]
    L --> M["Apply priority terrain / ability, Protect, Substitute-status and move-specific routing gates"]
    M --> N["Record hit outcome and recipient; return branch and operands"]
    J --> O["lax.switch branch"]
    N --> O
    O -->|"0"| P["do_damaging_move"]
    O -->|"1: ordinary status"| Q["_status_move_effects"]
    O -->|"6: Whirlwind status hit"| W["Defer forced switch"]
    O -->|"2"| R["do_stat_boost_from_move"]
    O -->|"3"| S["do_flash_fire_from_move"]
    O -->|"4"| T["do_healing_from_move"]
    O -->|"5"| U["is_immune"]
    P --> X["Prepare local forced-switch request if deferred"]
    Q --> X
    R --> X
    S --> X
    T --> X
    U --> X
    W --> X
    X --> Y["Commit enabled request: departure, hazards and entry hooks"]
    Y --> V["Set switch_requested when move flag and original branch permit"]
```

`effective_move` computes transient type, power and priority changes without
rewriting the move slot. Accuracy accounts for the represented stages, items,
abilities and weather; selected absorption/protection paths bypass an ordinary
miss. `_prepare_normal_hit` also records protection and Substitute status blocks,
including represented Spiky Shield retaliation. A damage hit's Substitute routing
is handled later, in `resolve_damaging_hit`.

Whirlwind reaches the deferred branch only after normal routing selects a
status hit. Its preparation handles drag immunity, ability reveals and random
replacement selection; commitment runs after the hit switch and before attempt
finalization. The request is local to this call, with no pending battle-state
field. Direct `do_status_move` callers prepare and commit immediately.

## 6. Damage and per-hit effects

```mermaid
flowchart TD
    A["do_damaging_move -> resolve_multi_hit"] --> B["Choose hit count; identify Beat Up contributors"]
    B --> C["Fixed 10-iteration lax.scan"]
    C --> D{"Hit still eligible?"}
    D -->|"No"| C
    D -->|"Yes"| E{"Later hit requires accuracy?"}
    E -->|"Yes: miss"| F["Record MISS; stop subsequent attempts"]
    E -->|"No or hit"| G["Set per-hit power; resolve_damaging_hit"]
    G --> H["Base power; crit and random roll; stage/stat selection"]
    H --> I["Base damage; weather, crit, random, STAB, type and other modifiers"]
    I --> J["Post modifiers, screens, held items; replace with fixed damage when applicable"]
    J --> K{"Substitute intercepts?"}
    K -->|"Yes"| L["Damage Substitute; record recipient and actual loss"]
    K -->|"No"| M["take_damage_value: HP, survival, item/ability and faint bookkeeping"]
    L --> N["Drain and single-hit recoil from recorded actual loss"]
    M --> N
    N --> O["If Pokémon hit: after_every_hit, contact/ability effects, balloon, held-item and Knock Off hooks"]
    O --> P["If Pokémon hit: secondary_effects; then applicable after_damage callbacks"]
    P --> C
    F --> C
    C -->|"Scan complete"| Q["Scale Shot stages; roll shared self-effect chance"]
    Q --> R["finish_damaging_move: recharge, self stages, after_move items, after_move_finished, self-KO / winner"]
    R --> S["Return to dispatch and step_move"]
```

The scan has a fixed size even for a single hit. Initial accuracy is checked
before the scan; some multi-hit moves check later hits independently. The
accumulated `damage` is calculated damage, while `HitResults` records actual
loss for an individual hit. Multi-hit recoil aggregation by total actual HP loss
is not implemented. Only certain `after_damage` callbacks also run after a
Substitute hit. See [passive mechanics](random-battle-passives.md) for supported
ability/item hooks and their limitations.

## 7. Status branch and turn aftermath

```mermaid
flowchart TD
    A["Status move"] --> B{"Move ID"}
    B -->|"Whirlwind"| C["Prepare forced switch: ability / bench check; choose legal replacement"]
    B -->|"Rest / Pain Split"| D["direct_status_move: direct status or HP assignment"]
    B -->|"Memento"| E["Apply opposing stat drops when hit reached; self-faint"]
    B -->|"Trick / Switcheroo"| F["swap_items: validate and atomically exchange; activate received items"]
    B -->|"Destiny Bond"| G["use_destiny_bond: activate or fail consecutive use"]
    B -->|"Substitute"| H["create_substitute: spend HP and store doll HP"]
    B -->|"Worry Seed"| I["worry_seed: validate and replace ability; lifecycle hooks"]
    B -->|"Other"| J["_generic_status_move"]
    J --> K["Protect chain; choose target; healing, status, stat changes"]
    K --> L["Field status and move restrictions; weather, terrain, Perish Song, hazards"]
    L --> M["Record represented failure conditions"]
    C --> C1["Commit switch after hit dispatch; immediate for direct API callers"]
    C1 --> N["Return to dispatch finalization"]
    D --> N
    E --> N
    F --> N
    G --> N
    H --> N
    I --> N
    M --> N
    N --> O["Scheduler finishes queue; step_field once"]
    O --> P["Side conditions; early residual ability cures"]
    P --> Q["Weather, terrain, Bad Dreams and status HP effects"]
    Q --> R["Leftovers; Moody; late residual abilities; orbs; item updates"]
    R --> S["Perish Song; decrement field timers; clear per-turn fields"]
    S --> T["refresh_paradox and resolve_faints; request replacements or terminal timestep"]
```

The status diagram groups individual generic callbacks; their actual ordering is
the order shown. `step_field` has known limitations around residual ordering and
unrepresented interactions. A move's immediate faint processing and a field
faint are separate from the environment's later terminal reward calculation.
