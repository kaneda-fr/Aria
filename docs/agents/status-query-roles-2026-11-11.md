# Noms, prédicats et pièces — quatrième revue du 2026-11-11

La quatrième revue autorisée de la [PR #5](https://github.com/kaneda-fr/Aria/pull/5),
`5854619410` sur `b6460a0`, expose C6 : une question générale sur les volets ouverts
pouvait sélectionner seulement un équipement nommé « Open », puis effacer le
prédicat. Le même mécanisme est reproduit pour les lumières nommées On/Off.
Ce défaut complète [C5](status-cover-name-review-2026-11-11.md), sans annuler sa correction.

## Correction

`app/plugins/inventory/query_roles.py:status_query_scope` sépare les noms littéraux,
les opérateurs d’état/nombre/quantification et les pièces dans les deux résolveurs.
Un mot fonctionnel ne devient un équipement que si la syntaxe le désigne : nom
explicite, sujet après une métrique, prédicat suivant, ou liste singulière.
Les questions générales gardent leur portée. Les noms de pièce restent littéraux,
y compris un mot d’état et une pièce inconnue ; aucun élargissement implicite.
Une portée ambiguë demande une précision avant toute lecture.
Les ellipses de pièce rétablissent les équipements de cette pièce ; les choix
ordinaux continuent de valider les options affichées. Aucun modèle, dépendance,
appel externe, changement d’historique, de voix ou de calibration.

## Validation et qualité

50 variantes ont été observées RED avant correction : 26 volets et 24 lumières /
noms d’opérateurs. Les **184 tests ciblés, 943 locaux et 921 serveur** passent.
Les 22 cas Mac exclus sur Linux passent avec mocks localement. Les fixtures
utilisent des équipements synthétiques ; aucun nom réel ni mapping privé publié.

Le dernier build passe **40 requêtes FR/EN** avec Jeedom réel : 22 lectures aux IDs
exacts, huit commandes exactes log-only, quatre couples clarification/reprise,
quatre questions générales et deux énumérations. Deux contrôles de file en plus.
Aucune erreur brute ni après audit indépendant des valeurs, langues et voix.
Les lectures n’utilisent pas de génération de modèle au premier plan.

Sur les 40 réponses communes au build `3d52f6d` (avant C5), 34 sont identiques.
Six réponses de volets passent de « fermé » à « position indéterminable » : les
observations Jeedom ont changé hors du seul endpoint fermé configuré. L’audit
indépendant applique le mapping privé aux valeurs reçues et confirme cet inconnu.
Cela ne démontre ni un endpoint ouvert ni une régression de sélection. Le volet
non répondant reste indisponible, conformément à la confirmation du mainteneur.

## Latence sur le serveur cible

| Même groupe de 7 lectures/langue | Avant C5/C6 (`3d52f6d`) | Après C6 |
| --- | ---: | ---: |
| Médiane admission → premier WAV FR | 1,092 s | 1,088 s |
| Maximum FR | 1,335 s | 1,249 s |
| Médiane EN | 1,140 s | 1,073 s |
| Maximum EN | 1,236 s | 1,451 s |
| File FR | 4,824 s | 4,805 s |
| File EN | 4,483 s | 4,380 s |

Ces passages uniques ne garantissent pas une absence de régression statistique.
Le maximum anglais augmente de 216 ms ; les réponses de volets ne sont pas toutes
identiques et leur synthèse reste variable. Le [diagnostic Piper/préchargement](status-live-and-tts-2026-11-11.md)
reste valable, sans changement de réglage pour masquer ce compromis.

Profil séparé du seul calcul de plan, cache identique, aucun runtime concurrent :
14 textes de lecture × 30 répétitions = 420 mesures par build. Avant/après C6 :
médiane **1,768 → 1,869 ms**, p95 **5,647 → 5,843 ms**, maximum **5,871 → 6,403 ms**.
Ce profil exclut historique, I/O, modèle et TTS ; il mesure le coût du parseur,
pas la latence d’une conversation. Aucun envoi réseau dans les deux profils.

## Audit et livraison

46 admissions / 46 issues terminales, deux rejets de surcharge attendus.
43 destinations réseau d’inférence, toutes loopback ; le pont réel ne publie que
71 info-get autorisés, sans payload ni retain. Les 104 fichiers Python produit
correspondent par hash entre source locale, copie d’essai et serveur. Six modules
du wheel correspondent exactement au code testé ; la configuration privée est exclue.
Les services sont arrêtés, aucun micro/haut-parleur réel ni action d’équipement.

C6 est destiné à la PR existante, avec réponse et résolution après validation/push.
Le mainteneur a autorisé un quatrième cycle (« Ok +1 »), désormais terminé.
Les sept défauts sont corrigés ; une revue du nouveau head reste nécessaire.
Aucun cinquième cycle, fusion, déploiement de production ou audio réel autorisé.
Les [rapports de revue](https://github.com/kaneda-fr/Aria/pull/5) portent l’état courant.
Ces essais API → WAV ne prouvent ni l’acoustique/ASR réel, ni l’identité d’une personne,
ni la position ouverte, ni l’achèvement physique d’une commande.
