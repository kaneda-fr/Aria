"""Speak inventory evidence without a second model pass or implied actuation."""

CAPABILITIES = {
    'light_on': ('allumer', 'turn on'),
    'light_off': ('éteindre', 'turn off'),
    'light_slider': ('régler la luminosité', 'adjust brightness'),
    'flap_up': ('ouvrir', 'open'),
    'flap_down': ('fermer', 'close'),
    'flap_stop': ('arrêter', 'stop'),
    'flap_slider': ('régler la position', 'adjust position'),
}


def render_lookup_results(results: list[dict], *, language: str) -> str:
    """Describe bounded search results, not physical state or completed actions.

    Input wording remains the model's responsibility. This renderer depends only
    on provider evidence, never particular user sentences or benchmark devices.
    A result limit must not imply that the displayed devices/controls are exhaustive.
    """
    fr = language == 'fr'
    groups = {}
    incomplete = failed = empty = False
    for result in results:
        if not result.get('success'):
            failed = True
            continue
        candidates = result.get('candidates', [])
        empty |= not candidates
        incomplete |= result.get('total_found', len(candidates)) > len(candidates)
        for candidate in candidates:
            # Same-named devices must not be merged into one apparent match.
            key = candidate.get('device_id')
            if not key or not candidate.get('device_name'):
                incomplete = True
                continue
            group = groups.setdefault(key, {'name': candidate['device_name'],
                                           'room': candidate.get('room'), 'caps': []})
            capability = candidate.get('capability')
            label = CAPABILITIES.get(capability, ('autres commandes', 'other controls'))[0 if fr else 1]
            if label not in group['caps']:
                group['caps'].append(label)

    if not groups:
        if failed:
            return ("L'inventaire est indisponible ; je ne peux pas vérifier les appareils."
                    if fr else "The inventory is unavailable; I can't check the devices.")
        if incomplete:
            return ("Les résultats ne permettent pas d'identifier les appareils. Précise le nom et la pièce."
                    if fr else "The results don't identify the devices. Please specify the name and room.")
        return ("Aucun appareil ne correspond à cette recherche. Précise le nom et la pièce."
                if fr else "No device matches this search. Please specify the name and room.")

    lines = []
    for group in list(groups.values())[:3]:
        name = group['name']
        if group['room']:
            name += f" ({group['room']})"
        caps = group['caps']
        incomplete |= len(caps) > 4
        labels = ', '.join(caps[:4])
        lines.append((f"Commandes disponibles pour {name} : {labels}." if fr else
                      f"Available controls for {name}: {labels}."))
    if incomplete or len(groups) > 3:
        lines.append("Résultats partiels ; précise l'appareil ou la pièce." if fr else
                     "These are partial results; narrow the device or room.")
    if failed:
        lines.append("Une recherche a échoué." if fr else "One search failed.")
    if empty:
        lines.append("Une recherche n'a donné aucun résultat." if fr else "One search had no matches.")
    lines.append("État actuel inconnu ; aucune action effectuée." if fr else
                 "Current state unknown; no action taken.")
    return ' '.join(lines)
