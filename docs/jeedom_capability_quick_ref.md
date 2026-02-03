# Jeedom Capability Quick Reference

> **Quick lookup for Jeedom `generic_type` → Aria capability mapping**
> Based on MQTT discovery analysis 2026-01-30

---

## Lighting Capabilities

| Jeedom `generic_type` | Aria Tag | Type | SubType | Value | Description |
|-----------------------|----------|------|---------|-------|-------------|
| `LIGHT_SLIDER` | `cap:light_slider` | action | slider | 0-99 | Set brightness level |
| `LIGHT_STATE` | `cap:light_state` | info | numeric | 0-100 | Current brightness |
| `LIGHT_ON` | `cap:light_on` | action | other | (fixed) | Turn on (full brightness) |
| `LIGHT_OFF` | `cap:light_off` | action | other | (fixed) | Turn off |

**Voice Patterns (FR):**
- "allume la lumière {room}"
- "éteins {device}"
- "mets la lumière à {value}%"
- "diminue/augmente la lumière"

**Voice Patterns (EN):**
- "turn on the light in {room}"
- "turn off {device}"
- "set {device} to {value}%"
- "dim/brighten the light"

---

## Shutter/Blind Capabilities (Flaps)

| Jeedom `generic_type` | Aria Tag | Type | SubType | Value | Description |
|-----------------------|----------|------|---------|-------|-------------|
| `FLAP_SLIDER` | `cap:flap_slider` | action | slider | 0-99 | Set position (0=closed, 99=open) |
| `FLAP_STATE` | `cap:flap_state` | info | numeric | 0-99 | Current position |
| `FLAP_UP` | `cap:flap_up` | action | other | (fixed) | Open/raise |
| `FLAP_DOWN` | `cap:flap_down` | action | other | (fixed) | Close/lower |
| `FLAP_STOP` | `cap:flap_stop` | action | other | (fixed) | Stop movement |

**Voice Patterns (FR):**
- "ouvre/ferme les volets {room}"
- "monte/descend {device}"
- "arrête les volets"
- "mets les volets à {value}%"

**Voice Patterns (EN):**
- "open/close the shutters in {room}"
- "raise/lower {device}"
- "stop the shutters"
- "set {device} to {value}%"

**Observability Notes:**
- Position changes are **asynchronous** (motor movement takes time)
- Typical timeout: 30-60s for full open/close
- Use `predicate: "equals"` for exact position matching
- Power consumption (`POWER`) can be secondary confirmation (motor running → power > 0)

---

## Energy Monitoring Capabilities

| Jeedom `generic_type` | Aria Tag | Type | SubType | Unit | Description |
|-----------------------|----------|------|---------|------|-------------|
| `POWER` | `cap:power` | info | numeric | W | Instantaneous power consumption |
| `CONSUMPTION` | `cap:consumption` | info | numeric | kWh | Cumulative energy consumption |

**Usage:**
- Typically attached to light and shutter devices
- Info-only (not executable)
- Can be used for energy monitoring queries: "combien consomme {device}?"

---

## Generic/Technical Capabilities (Usually Filtered)

| Jeedom `generic_type` | Aria Tag | Type | SubType | Description |
|-----------------------|----------|------|---------|-------------|
| `GENERIC_INFO` | `cap:generic_info` | info | * | Generic sensor data |
| `DONT` | (skip) | * | * | Explicitly ignored |
| `null` | (skip) | * | * | Unmapped/technical commands |

**Common Unmapped Commands (Z-Wave):**
- `0-0-pingNode`: Network ping
- `0-0-healNode`: Network repair
- `0-0-nodeStatus`: Node status string
- `0-0-isFailedNode`: Node health check

**Recommendation:** Filter out during normalization unless explicitly needed for diagnostics.

---

## Domain Tags (from `category`)

| Jeedom Category | Aria Tag | Typical Capabilities |
|-----------------|----------|----------------------|
| `light` | `domain:light` | LIGHT_SLIDER, LIGHT_ON, LIGHT_OFF |
| `opening` | `domain:opening` | FLAP_SLIDER, FLAP_UP, FLAP_DOWN |
| `heating` | `domain:heating` | THERMOSTAT_*, TEMPERATURE |
| `energy` | `domain:energy` | POWER, CONSUMPTION |
| `security` | `domain:security` | ALARM_*, LOCK_*, PRESENCE |
| `multimedia` | `domain:multimedia` | VOLUME, PLAY, PAUSE |

---

## Intent → Capability Mapping (Hot Path)

### Close Shutters
**Input:** "ferme les volets de la salle à manger"
**Intent Tags:** `cap:flap_down`, `cap:flap_slider`, `domain:opening`
**Location:** `room:rdc`
**Candidates:**
1. `cmd_id=5453` (FLAP_DOWN) → score 0.95 (exact match)
2. `cmd_id=5450` (FLAP_SLIDER, value=0) → score 0.85 (can achieve same result)

**Selected:** `5453` (highest score, simpler action)

### Set Light to 50%
**Input:** "mets la lumière à 50 pourcent"
**Intent Tags:** `cap:light_slider`, `domain:light`
**Value:** `50`
**Location:** (inferred from context or follow-up)
**Candidates:**
1. `cmd_id=5630` (LIGHT_SLIDER, Lumiere Salle a Manger New) → score 0.90

**Selected:** `5630` with `value=50`

### Turn Off All Lights in Room
**Input:** "éteins toutes les lumières du salon"
**Intent Tags:** `cap:light_off`, `domain:light`
**Location:** `room:salon`
**Candidates:**
1. `cmd_id=5632` (LIGHT_OFF, Lumiere Salle a Manger New)
2. `cmd_id=5587` (LIGHT_OFF, Lumiere Sofa)
3. ... (all lights in room)

**Selected:** Multiple commands (parallel execution)

---

## Execution Timeout Recommendations

| Capability | Typical Timeout | Notes |
|------------|-----------------|-------|
| `light_on`, `light_off` | 3-5s | Fast response (Z-Wave) |
| `light_slider` | 3-5s | Fast response |
| `flap_up`, `flap_down` | 30-60s | Full open/close movement |
| `flap_slider` | 10-60s | Depends on distance to target |
| `flap_stop` | 2-3s | Immediate command |
| `thermostat_setpoint` | 5-10s | State update delay |

**Predicates:**
- `equals`: Exact value match (on/off, specific position)
- `within_tolerance`: Numeric value with margin (dimming, temperature)
- `changed`: Any state change (fallback for no target)

---

## Multilingual Support (EN/FR)

### French Keywords
| Intent | Keywords |
|--------|----------|
| Light On | allume, ouvre lumière |
| Light Off | éteins, ferme lumière |
| Dim | diminue, baisse |
| Brighten | augmente, monte |
| Shutter Open | ouvre, lève, monte |
| Shutter Close | ferme, baisse, descend |
| Stop | arrête, stop |

### English Keywords
| Intent | Keywords |
|--------|----------|
| Light On | turn on, switch on |
| Light Off | turn off, switch off |
| Dim | dim, lower, decrease |
| Brighten | brighten, raise, increase |
| Shutter Open | open, raise, lift |
| Shutter Close | close, lower |
| Stop | stop, halt |

**Implementation:** Use multilingual packs (see spec) with token-based matching, not LLM for hot path.

---

## Edge Cases & Gotchas

### 1. Value Template Handling
- `#slider#`: Replace with user-provided value
- `true` / `false`: Boolean for Z-Wave commands
- `99` / `0`: Fixed values (light on = 99, off = 0)
- `255`: "Restore last value" (special Z-Wave command)

### 2. Observer Linkage
- Action commands often have `"value": "5454"` pointing to observer cmd_id
- Use this for automatic expectation creation
- If missing, execution is optimistic (no confirmation)

### 3. Range Normalization
- Jeedom light sliders: 0-99 (not 0-100!)
- Jeedom shutter sliders: 0-99 (0=closed, 99=open)
- State reporting may show 0-100 for percentage display
- Always check `minValue` / `maxValue` in configuration

### 4. Power Consumption as Proxy
- Some devices don't report state quickly
- Power > 0 often means "device active/moving"
- Useful for secondary confirmation but not primary expectation

---

## Testing Checklist

- [ ] Light on/off (binary)
- [ ] Light dimming (slider, exact value)
- [ ] Light dimming (slider, tolerance)
- [ ] Shutter full open
- [ ] Shutter full close
- [ ] Shutter position (slider, exact)
- [ ] Shutter stop mid-movement
- [ ] Multi-device execution (all lights in room)
- [ ] Timeout handling (unresponsive device)
- [ ] Power monitoring confirmation
- [ ] Multilingual intent matching (EN/FR)
- [ ] Room disambiguation (multiple devices same name)

---

**Ready for implementation and testing.**
