# Contrat proposé pour les faits Jeedom

2026-10-21 — Annexe de l’[étude](jeedom-status-semantics-2026-10-21.md).
**Non implémenté** ; données d’exemple fictives. Les règles sont une proposition
locale sans modèle supplémentaire, compatible avec les deux stratégies de contrôle.

2026-11-10 : le [pilote lumières/volets](status-pilot-2026-11-10.md) implémente un
sous-ensemble de cette proposition ; les autres domaines et extensions restent à étudier.

## Contrat proposé : définition, observation, fait

**Définition de propriété**, relativement stable, compilée au chargement et au
changement d’inventaire :

- identité composée `provider + equipment_id + command_id`, identité du canal et
  version d’inventaire ; label et emplacement séparés de l’identité ;
- propriété canonique : `light.on`, `light.level`, `cover.open_percent`,
  `temperature`, `rain.probability`, `rain.amount`, `wind.speed`, `wind.direction` ;
- type, unité native/canonique, conversion documentée, plage, valeurs énumérées,
  polarité, résolution et éventuelles valeurs sentinelles ;
- nature : mesure, consigne, état logique, compteur, total de période, prévision,
  diagnostic ; périmètre physique et période de validité ;
- origine de l’interprétation : type générique Jeedom conservé, adaptateur documenté
  du fournisseur/modèle, ou réglage local validé ; version de règle.

Ordre proposé : correction locale explicitement validée, mapping fournisseur vérifié,
métadonnées Jeedom compatibles. Choix proposé par l’étude : une correction locale
résout un désaccord documenté, attaché à l’identité et à l’empreinte des métadonnées
concernées ; ce désaccord attendu ne bloque pas l’interprétation. Un changement de
source, de type, d’unité, de polarité ou de canal invalide cette correction jusqu’à
revalidation. Un conflit nouveau ou non résolu bloque l’affirmation et remonte un
diagnostic ; une règle générique ne masque pas une contradiction. Les mots du nom
et les unités peuvent suggérer un mapping à examiner, jamais créer seuls une polarité,
une source préférée, une pièce ou une période. Préserver les métadonnées inconnues
utiles dans un bloc fournisseur local plutôt que les perdre à la normalisation.

**Observation**, indépendante de la définition : valeur native, canal/source,
horodatages de réception, récupération, collecte et changement si disponibles ;
statut de lecture, provenance, version/séquence lorsqu’elle existe. Conserver `null`
et la raison d’absence plutôt qu’inventer une heure. Une valeur en cache reste une
dernière valeur connue après déconnexion, jamais une observation redevenue fraîche.

**Fait dérivé** : propriété normalisée, valeur typée, références aux observations,
règle/version appliquée, disponibilité, fraîcheur et limites. Garder ces dimensions
séparées : une valeur reçue peut être valide mais ancienne, ou de date inconnue.
Une probabilité numérique de confiance serait trompeuse sans calibration ; utiliser
des raisons explicites (`mapping_missing`, `timestamp_missing`, `source_conflict`).

Le cœur peut rester en dataclasses et dictionnaires indexés. Ni base vectorielle,
ni embeddings, ni entraînement requis pour interpréter une valeur déjà identifiée.

## Règles par domaine

| Domaine | Faits utiles | Ce qu’il ne faut pas conclure sans preuve |
| --- | --- | --- |
| Lumières | état de chaque circuit ; niveau de variation si son échelle est connue ; équipement partiellement allumé | un relais = une ampoule ; puissance positive = lumière physiquement allumée ; un canal inconnu = éteint |
| Volets | position d’ouverture normalisée, ouvert/fermé/partiellement ouvert ; mouvement si une source le rapporte | 0/100 ont partout le même sens ; deux positions identiques prouvent l’arrêt ; une consigne est une position mesurée |
| Chauffage | température mesurée, consigne, mode, activité de chauffe séparés | une consigne atteinte prouve l’arrêt du chauffage ou le confort |
| Contacts/présence | ouvert/fermé ou présence détectée, selon la polarité déclarée | `0/1` est universel ; absence de mouvement = maison inoccupée |
| Météo | observation/prévision, lieu, période, température, humidité, vitesse/direction, cumul/probabilité distincts | les mm sont un débit ; un pourcentage prouve qu’il pleut ; la prévision de demain décrit maintenant |
| Énergie/eau | puissance instantanée, index cumulé, consommation d’une période, volume et coût distincts | W = Wh ; m³ = kWh ; total du compteur = consommation du jour |

Pour les volets, choisir **en interne** 0 = fermé, 100 = ouvert est une convention
possible, comme celle de [Home Assistant Cover](https://contributors.home-assistant.io/core/entities/cover/).
Cela ne dit rien du sens des données Jeedom. La conversion exige une convention
fournisseur ou un réglage local vérifié. Ni `invertBinary` d’affichage ni une plage
d’action ne suffisent, seuls, à prouver la signification de la valeur d’information.
Ne pas borner artificiellement une valeur invalide pour la rendre plausible.

Un équipement à deux circuits peut devenir `partly_on`, avec les deux faits sources.
Un alias utile pour retrouver un appareil ne définit pas son câblage. Les identités
techniques restent disponibles pour le diagnostic, sans devenir le vocabulaire oral.

### Vocabulaire destiné à l’utilisateur

Décision du mainteneur, 2026-11-10 : les réponses utilisent les mots du quotidien.
Dire « une lumière sur deux est allumée », « le volet est à moitié ouvert » ou
« je ne connais pas son état », selon les faits disponibles. En anglais, appliquer
la même règle : « one of the two lights is on ». Éviter « circuit », « canal »,
« relais », « état binaire » et les noms de propriétés dans les réponses ordinaires.
Les noms familiers validés de la pièce ou de la lumière restent utilisables.

Le choix des mots ne change pas les comptes : dire « une lumière sur deux » lorsque
ces deux éclairages sont bien identifiés. Si seule une partie technique est connue,
dire « une partie des lumières est allumée », sans inventer un nombre d’ampoules ni
un nom comme « plafond » ou « canapé ». Rendre aussi les limites compréhensibles :
« je ne sais pas si ces informations sont à jour ». Garder codes, canaux, provenance
et détails de calcul dans les logs ou dans une demande explicite de diagnostic.

## Temps, fraîcheur et exhaustivité

Le [code officiel Jeedom](https://github.com/jeedom/core/blob/master/core/class/cmd.class.php)
expose notamment `getCollectDate()` et `getValueDate()`. Leur présence en PHP ne
prouve pas leur transmission dans notre flux MQTT, ni une mesure physique récente.
La distinction collecte/changement et la convention de fuseau doivent être vérifiées
sur la version installée avant exploitation. Un état inchangé peut être valide ;
la date du dernier changement ne constitue pas seule une durée de péremption.

Le pont actuel reçoit une valeur sans date de mesure. Première version : conserver
`freshness=unknown` et répondre honnêtement. Pour mieux faire ensuite, étudier une
enveloppe de lecture locale versionnée dans notre plugin Jeedom avec valeur,
métadonnées temporelles et identifiant de requête. Ne pas modifier MQTT Manager,
introduire une clé admin ou réintroduire des signatures non demandées. Cette extension
du plugin, présente dans le checkout canonique séparé, reste une option à valider.

Fraîcheur par propriété/profil : délai attendu explicite, origine du délai et instant
d’évaluation. Sans cadence documentée, ne pas inventer un seuil universel de cinq
minutes. Les diagnostics de connexion décrivent le transport, pas chaque capteur.
L’acquisition amont d’une météo reçue sur le LAN doit également être documentée :
une adresse locale ne prouve pas que le fournisseur amont fonctionne hors Internet.

Un rapport doit porter deux couvertures :

1. **Lecture** : propriétés attendues, effectivement lues, indisponibles, non interprétées
   et éventuellement non demandées à cause du budget.
2. **Inventaire** : équipements connus et complétude du catalogue, actuellement inconnue.

Ainsi « les quatre éclairages répertoriés sont rapportés éteints » est justifiable
si leurs cinq états le permettent ; « toute la maison est éteinte » ne l’est pas.
Compter équipements et circuits séparément. Un témoin positif suffit pour « au moins
un allumé » ; une affirmation universelle exige toute la couverture de la portée
annoncée. Sinon préciser le minimum connu et l’incertitude.

Plusieurs lectures ne sont pas un instantané physique atomique. Conserver leur
intervalle temporel et signaler l’incohérence si l’écart est excessif pour la propriété.
Événements répétés, retardés ou retenus ne doivent pas réinitialiser artificiellement
la fraîcheur. Sans séquence ou horloge source, ne pas prétendre connaître leur ordre réel.

## Paquet destiné au LLM et parole

Exemple de contrat **proposé**, données entièrement fictives. Les clés canoniques
restent indépendantes du français/anglais ; les labels sont des données.

```json
{
  "schema": "aria.status/1",
  "scope": {"domain": "light", "room": "room-A", "catalog_complete": false},
  "coverage": {"expected_channels": 2, "read_channels": 2, "interpreted_channels": 2},
  "facts": [{
    "entity": "jeedom:equipment-A",
    "label": "Éclairage pièce A",
    "property": "light.aggregate",
    "value": "partly_on",
    "on_channels": 1,
    "off_channels": 1,
    "unknown_channels": 0,
    "freshness": "unknown",
    "observed_at": null,
    "evidence_refs": ["observation-A1", "observation-A2"],
    "rule": "light.aggregate/1"
  }],
  "limitations": ["measurement_time_unknown", "catalog_completeness_unknown"],
  "actions_performed": []
}
```

Les observations référencées sont résolubles dans le registre local, avec valeurs,
horodatages et métadonnées source ; elles ne doivent pas toutes être dupliquées dans
le prompt. Le paquet réel inclut aussi version d’inventaire, instant d’évaluation et
identifiant de requête. Les agrégats invalidés par un changement de source, d’unité,
de canal ou de règle sont recalculés ; aucun fait ne survit silencieusement à ce changement.

Sélectionner avant génération selon propriété, pièce, période et demande. Envoyer
les exceptions et quelques faits pertinents, pas les 2 383 commandes. Tout dépassement
est explicite : réduire la portée par clarification, jamais tronquer puis dire « tous ».
Une source de température parmi plusieurs exige un rôle/emplacement ou une priorité
locale déclarée ; aucune moyenne ou préférence arbitraire. Même discipline pour
doublons, virtuels et compteurs parents/enfants afin d’éviter les doubles comptes.

Réponse simple : rendu FR/EN à partir des faits, **zéro appel modèle**, avec synthèse
courte puis détails sur demande. Réponse composée : le LLM existant reçoit ce paquet
par le chemin d’outils partagé ; aucun modèle chargé de nettoyer chaque événement.
Le surcoût éventuel de génération d’une réponse composée reste à mesurer. Les trois
échanges d’historique et les journaux complets sont conservés ; les faits présents
ne se confondent pas avec une ancienne réponse dans l’historique.

Le serveur reste responsable du calcul, des quantificateurs, des nombres et des
limites. Une instruction de prompt ne garantit pas que le modèle les respectera.
Pour le pilote, réserver les assertions d’état au rendu contrôlé ; laisser au modèle
la conversation générale. Une future verbalisation libre demanderait sa propre
évaluation et un repli déterministe, sans nouvelle génération en cas de doute.
Traiter les noms et valeurs textuelles comme données, jamais comme instructions.
