# Calibration locale « fermé uniquement » — 2026-11-10

Le mainteneur confirme que les volets sont actuellement fermés et demande une
configuration en fichier, pas des constantes propres à la maison dans le code.
Une lecture Jeedom réelle via le pont read-only obtient deux valeurs égales à zéro
sur trois propriétés de position. La troisième ne répond pas, y compris après une
nouvelle lecture ciblée : aucune valeur ne lui est attribuée par analogie. Le mainteneur
confirme ensuite que cette absence de réponse est attendue ; elle ne constitue pas
un échec de calibration à investiguer. Son état reste indisponible.

Deux entrées liées à l’identité exacte sont enregistrées dans un JSON local ignoré
par Git, copié dans la configuration privée du serveur, mode 0600. Le sélecteur
`ARIA_STATUS_MAP_FILE` reste dans les fichiers d’environnement locaux. Les noms,
IDs et valeurs propres à l’installation ne sont pas ajoutés aux sources publiques.
Chaque entrée utilise `calibration: "closed_only"` et `closed_value: 0`, sans borne
ouverte. La plage 0–99 des commandes liées n’est pas prise pour une preuve de
position ouverte. Ce nouveau mode générique reconnaît seulement la valeur fermée
confirmée ; les autres valeurs restent inconnues. Une configuration contradictoire
avec une borne ouverte est refusée. Les lectures indisponibles et changements
d’identité restent protégés. Aucun mouvement, rafraîchissement de capteur ou audio.

Cinq cas de régression supplémentaires vérifient fichier réel, polarités inverses,
valeurs non calibrées, indisponibilité, identité et configuration contradictoire,
avec un cycle RED→GREEN. Les suites ciblées passent 76 tests. La suite serveur
complète passe **807 tests**. Localement : 828 passes et un échec du test temporel
`test_heartbeat_renews_lease_during_long_clip` ; sa relance isolée passe. Il dépend
d’une échéance de 20 ms et n’utilise pas le nouveau module. Ce résultat n’est pas
présenté comme une suite locale entièrement verte ni une cause de latence prouvée.
GitNexus classe `_interpret` HIGH, appelé par `project_status`, en amont des quatre
parcours de lecture lumière/volet et individuels ; cette portée a été annoncée.

Le serveur recharge le fichier privé et projette les réponses réellement reçues :
deux fermés et un inconnu, rendus cohérents FR/EN. Des valeurs différentes laissées
sans calibration produisent toutes « inconnu ». Le premier chargement et les deux
rendus prennent environ 3,55 ms ; ce contrôle n’est pas une campagne E2E de latence.
Les outils restent log-only et Aria est toujours arrêtée. Le fichier d’environnement
privé serveur doit être chargé au prochain lancement du pilote ; aucun redémarrage
ni déploiement du service réel n’a été demandé. Les preuves sont conservées dans
`.local/status-pilot/cover-config/` du checkout canonique.

[Suivi du parcours réel et de la préparation de PR](status-live-and-tts-2026-11-11.md).
