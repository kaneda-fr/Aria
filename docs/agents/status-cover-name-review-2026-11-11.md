# Noms de volets et prédicats d’état — revue du 2026-11-11

La troisième revue de la [PR #5](https://github.com/kaneda-fr/Aria/pull/5),
`5851504884` sur `3d52f6d`, annonce « No open issues » mais contient dans son corps
un défaut modéré non publié comme discussion : `cover_queries.py:64`.
Identifiant de suivi : `review:5851504884:body:1` (C5).

Une demande de statut d’un volet nommé « Open » pouvait sélectionner le prédicat
« ouvert », puis masquer un état réellement fermé. Le résolveur exclut maintenant
les segments correspondant aux noms sélectionnés avant d’extraire ouvert/fermé.
Le filtre explicite situé hors du nom reste interprété. Aucun modèle, dépendance,
paramètre d’historique/TTS ou changement de calibration n’est ajouté.

## Validation

Huit variantes FR/EN avec les noms Open, Closed, Ouvert et Fermé échouent avant
correction, puis passent dans le chemin partagé de requête jusqu’au rendu des
faits. Elles vérifient la lecture du bon ID info, l’état fermé configuré dans la
fixture et la conservation d’un prédicat explicite opposé hors du nom.
Les fixtures utilisent des noms et mappings synthétiques ; elles ne modifient pas
le catalogue Jeedom. 68 tests ciblés, **873 tests locaux et 851 tests serveur**
passent. Les 22 tests propres au client Mac passent avec mocks localement.

La validation réelle sur le serveur cible couvre **10 requêtes FR/EN**, huit
lectures aux IDs exacts, deux couples clarification/reprise et deux contrôles de
file supplémentaires. Deux requêtes nomment un équipement réellement calibré à
partir de l’inventaire. Les huit réponses communes restent identiques à la
[campagne précédente de 40 cas](status-live-and-tts-2026-11-11.md).
Aucune anomalie brute ou après audit indépendant de langue, voix et valeurs.
Cette campagne ciblée suit la correction C5 ; les 40 cas antérieurs couvrent
également lumières, questions générales, énumérations et huit commandes log-only
avant C5. Ils ne sont pas présentés comme 40 nouvelles exécutions du dernier build.

| Volets, même groupe de 3 lectures/langue | Avant C5 | Après C5 |
| --- | ---: | ---: |
| Médiane requête → premier WAV FR | 1,235 s | 1,111 s |
| Maximum FR | 1,335 s | 1,165 s |
| Médiane EN | 1,140 s | 1,223 s |
| Maximum EN | 1,159 s | 1,261 s |
| File FR | 4,824 s | 4,802 s |
| File EN | 4,483 s | 4,434 s |

Les deux lectures nommées donnent 1,036 s FR et 0,991 s EN. Le groupe anglais
augmente d’environ 83 ms en médiane et 102 ms au maximum ; le groupe français
baisse. Un passage par build ne prouve pas une absence de régression statistique.
Le diagnostic Piper/préchargement reste celui du rapport précédent ; aucun réglage
n’est changé pour masquer ce compromis.

L’audit compte 16 admissions/16 issues terminales, deux rejets de surcharge
attendus, 24 destinations réseau d’inférence toutes loopback et uniquement des
publications MQTT info-get sur le pont réel. Le paquet est reconstruit et ne
contient pas de configuration privée. Les services d’essai sont arrêtés après
collecte ; aucun micro/haut-parleur réel ni action d’équipement.

## Livraison et suite

Le quatrième cycle a été autorisé après ce résultat et a révélé C6, distinct de C5.
Voir la [correction partagée et les 40 nouveaux cas](status-query-roles-2026-11-11.md).
Les rapports de la [PR #5](https://github.com/kaneda-fr/Aria/pull/5) portent la décision
courante. Fusion/déploiement restent en attente ; aucun audio réel autorisé.
