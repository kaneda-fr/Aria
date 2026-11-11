# Statuts Jeedom réels et diagnostic TTS — 2026-11-11

Suite au [pilote lumières/volets](status-pilot-2026-11-10.md), le mainteneur demande,
dans cet ordre : parcours complet avec Jeedom réel, isolation du pic TTS anglais,
puis préparation de la PR. Les actions restent log-only ; aucun microphone ni
haut-parleur réel. La calibration demeure dans les fichiers privés ; voir
[les observations fermé uniquement](cover-calibration-2026-11-10.md).

## Lecture réelle et équipement indisponible

Deux lectures successives des trois propriétés volets reproduisent un défaut du
pont : la première rapporte deux états disponibles et un indisponible ; la seconde
répond `awaiting_previous_response` sans lire les équipements disponibles. La
quarantaine d’une commande expirée invalidait tout lot qui la contenait.

Le correctif garde cette quarantaine par commande. Une lecture groupée inclut les
états disponibles et une ligne indisponible pour la commande bloquée, sans la
republier. Une réponse tardive est toujours jetée avant d’autoriser un nouveau get.
Si toutes les commandes sont bloquées, le statut global reste compatible, avec
les lignes indisponibles demandées. Aucun action topic, retry temporel, timestamp de
mesure inventé ou changement de délai réseau n’est ajouté.

Trois nouveaux tests échouent avant le correctif et passent ensuite : répétition du
lot partiel, réponse tardive pendant un lot mixte, lot entièrement bloqué sans envoi.
Les 41 tests ciblés passent ; une nouvelle vérification réelle confirme que les
deux équipements disponibles sont lus aux deux appels. GitNexus retrouve l’identité
exacte de la méthode mais son risque reste UNKNOWN, sans arête du dispatch dynamique
`asyncio.to_thread` ; Serena/source confirment l’appel du point HTTP `/read`.

## Parcours complet FR/EN sur le serveur

Un serveur d’inférence isolé consomme le pont Jeedom réel par socket Unix. Seuls les
gets d’infos autorisés sont publiés ; le chemin actions n’a pas de client MQTT.
Les 28 cas passent, dont 14 lectures avec IDs exacts et faits comparés aux valeurs
reçues, deux clarifications puis reprises ordinales, quatre réponses générales et
huit commandes exactes enregistrées en log-only. Les lectures d’état n’utilisent
aucune génération par le modèle ; le préchargement spéculatif reste séparé.
Les voix FR/EN correspondent aux requêtes. L’état du volet sans réponse reste inconnu,
sans bloquer les autres ; il est confirmé attendu par le mainteneur.

| Admission → premier WAV disponible | Français, 7 lectures | Anglais, 7 lectures |
| --- | --- | --- |
| Médiane | 1,125 s | 1,148 s |
| Minimum | 0,464 s | 0,415 s |
| Maximum | 1,481 s | 1,305 s |

Ces valeurs incluent la lecture MQTT et la préparation du WAV, sans file concurrente.
Elles ne mesurent pas le début sonore physique. Les nombres ne sont pas une SLO ni
une comparaison avant/après avec le fournisseur synthétique du premier pilote.
L’inventaire reste non certifié complet et Jeedom ne fournit pas d’heure de mesure :
les réponses conservent ces limites au lieu de transformer la récupération en fraîcheur.
L’audit de l’inférence observe uniquement loopback, sans tentative externe/DNS ;
les publications du pont sont strictement des gets d’infos. Il n’exécute aucune action.

## Pic TTS reproduit et compromis du préchargement

Deux profils séquentiels utilisent le même code, inventaire synthétique contrôlé,
modèle, prompt, six messages d’historique, voix et paramètres. Ils rejouent exactement
les 18 tours français précédant le bilan anglais lent du pilote, avec livraison des
WAV et cache TTS séparé neuf par profil. Seul `ARIA_HISTORY_PREFILL` change. Une
instrumentation privée chronomètre l’entrée dans les threads et les processus Piper
et ffmpeg ; elle ne fait pas partie du code livré. Le système de fichiers de modèles
reste partagé et le cache disque du noyau n’est pas purgé.

| Mesure du tour anglais ciblé | Préchargement actif | Préchargement désactivé |
| --- | --- | --- |
| Admission → premier WAV | 2,460 s | 1,005 s |
| Attente du thread, maximum | 0,37 ms | 1,05 ms |
| Processus Piper | 2,247 s | 0,808 s |
| Dont chargement de voix, journal Piper | 0,519 s | 0,301 s |
| Dont inférence, journal Piper | 1,567 s | 0,399 s |
| Conversion ffmpeg | 0,140 s | 0,095 s |
| Tour de calcul français précédent → premier WAV | 2,647 s | 7,765 s |

Le texte du bilan est prêt en environ 20 ms et ne consomme aucune génération de
réponse. Le délai supplémentaire se situe principalement dans Piper. L’annulation
HTTP du préchargement précédent est journalisée avant la synthèse. La comparaison
contrôlée soutient une concurrence CPU liée au préchargement sur le Xeon ; le calcul
natif Ollama encore actif après annulation est l’explication probable. Sa durée
résiduelle n’est pas directement mesurée, et une seule paire ne quantifie pas une
SLO ou toute la variabilité de charge système.

Les **38 réponses** des deux profils sont identiques à la référence, voix correctes.
La désactivation gagne environ 1,45 s sur ce tour mais coûte environ 5,12 s au calcul
précédent : elle n’est pas retenue comme solution globale. Aucun changement au
modèle, à l’historique ou au comportement TTS/préchargement n’est livré. La suite
conseillée est de mieux borner le travail spéculatif encore actif à l’arrivée d’une
requête, en comparant aussi les questions générales ; éviter une règle propre au
nombre ou à la phrase du benchmark.

Les contrôles de file restent corrects dans les deux langues : premier WAV du tour
mis en file FR/EN 4,875/4,761 s avec préchargement, 4,847/4,473 s sans. Un essai par
langue et profil ne prouve pas une amélioration stable. Chaque profil compte 25
requêtes acceptées/25 issues terminales ; les deux rejets de surcharge attendus
restent visibles. Les audits réseau observent 35 puis 16 destinations loopback,
aucune destination externe ni tentative DNS.

## Validation et livraison

- `python -m pytest tests -q --tb=short` : **832 tests locaux réussis**, quatre
  avertissements existants. Le précédent test temporel instable passe dans ce run
  complet, sans modification de son échéance.
- Sur serveur Python 3.11 : **810 tests réussis**, quatre avertissements ; les 22
  tests propres au client Mac sont exécutés avec mocks dans la suite locale.
- Wheel construit sans dépendances ni réseau ; les nouveaux modules sont présents,
  et la configuration locale est absente de l’archive.
- SHA256 des 103 fichiers Python d’application/scripts du runtime vérifié contre
  l’envoi. Toutes les unités d’essai sont arrêtées, aucun client audio Mac relancé.
- Détection GitNexus avant commit : 28 fichiers, 186 symboles, 43 processus, risque
  CRITICAL sur le diff cumulé du chemin partagé. Portée annoncée et vérifiée.
- Les fichiers privés conservent observations, réponses exactes, traces, sources et
  manifestes dans `.local/status-next/`. Les fichiers de calibration ne sont pas
  ajoutés à la PR.

La PR préparée regroupe le correctif de sélection lumières hérité, le pilote de
faits partagés, la calibration partielle et l’isolation de la quarantaine. Elle ne
revendique pas la résolution du pic TTS ni un gain uniforme de latence. Le service
réel reste arrêté ; déploiement depuis main et essai vocal réel attendent la suite
explicitement demandée par le mainteneur. Aucun merge automatique.

## Corrections de revue avant livraison

La première revue Copilot du head `71b1a8e` confirme trois défauts modérés : choix
ordinal de lumière reposant sur des propositions recalculées, acceptation d’une
observation sans `device_id`, et affirmation globale/exacte malgré un catalogue
incomplet. Une relecture supplémentaire reproduit aussi une définition périmée dans
les deux variantes de lecture legacy. L’unité passe de % à W pendant l’attente,
mais l’ancien snapshot permettait encore un fait « allumée ».

Les corrections refusent les choix ordinaux lorsque les libellés affichés ont
changé ou sont indiscernables, exigent l’identité explicite de l’observation et
traitent le catalogue incomplet comme une incertitude pour les universels/comptes.
Les témoins existentiels positifs et contre-exemples valides restent utilisables.
La grammaire courte des choix anglais accepte les pronoms d’ordinal sans effacer
une pièce inconnue ou un nom d’équipement déclaré.

Les quatre lecteurs de statuts capturent les définitions avant l’attente, puis
vérifient le catalogue après réception. Des définitions modifiées, désactivées ou
impossibles à revérifier rendent la lecture indisponible. Les lecteurs legacy et
les chemins directs partagent ce contrôle ; météo et autres métriques conservent
leur parcours. Cela ajoute un contrôle local de catalogue, sans nouvelle génération,
ni publication MQTT supplémentaire de valeur/commande.

Seize cas supplémentaires sont observés RED avant correction, puis GREEN. Quatre
autres couvrent les choix inchangés et l’échec du contrôle de catalogue. La suite
complète passe désormais **852 tests locaux / 830 tests serveur**, quatre avertissements
existants. Les impacts projection/rendu sont HIGH ; le résolveur lumière est LOW.
Le graphe ne résout pas les dispatchs dynamiques des méthodes legacy ; leurs callers
outil LLM / lecture résolue et les deux lecteurs groupés sont vérifiés en source/tests.

Le dernier build est testé dans **34 cas réels FR/EN** : 18 lectures aux IDs exacts,
huit commandes log-only, quatre paires clarification/reprise et quatre réponses
générales. Les valeurs et états indisponibles sont vérifiés indépendamment, avec
voix et langue demandée correctes. Les 28 cas communs gardent 26 réponses identiques ;
les deux changements retirent le « oui » global non prouvé. Aucun changement de modèle,
d’historique, d’action réelle ou de traitement TTS/préchargement n’est introduit.

Le banc réorganisé contenait une erreur d’oracle : son contrôle TTS comparait chaque
cas à la dernière variable de groupe (`en`), générant **17 faux signalements français**.
Ils restent dans le rapport brut. Un audit séparé vérifie, pour chacun des 34 cas,
la langue demandée, la langue TTS et le fichier de voix réellement sélectionné :
aucune discordance réelle ni autre échec. Le script de reproduction est corrigé et
sa version exécutée est conservée. Il ne s’agit donc pas de 34 passes automatiques
sans aucun drapeau brut.

| Même groupe de sept lectures par langue | Avant revue | Après corrections |
| --- | --- | --- |
| Médiane FR | 1,125 s | 0,997 s |
| Maximum FR | 1,481 s | 1,189 s |
| Médiane EN | 1,148 s | 1,182 s |
| Maximum EN | 1,305 s | 1,401 s |

Le retrait du « oui » permet aussi une réutilisation de WAV identique pour un état
courant, ce qui influence les minima/médianes. Une exécution par build et langue ne
prouve pas un gain uniforme ; le maximum anglais augmente d’environ 95 ms dans ce
groupe. Les contrôles de file FR/EN passent à 4,761 / 4,451 s. Audit : 40 requêtes
acceptées/40 issues terminales, deux rejets de surcharge attendus, réseau d’inférence
loopback uniquement (37 destinations observées), et publications du pont info-get
uniquement. Les 103 fichiers Python de produit sont vérifiés par hash, les unités
d’essai sont arrêtées et aucun client audio réel n’est relancé.

Le diagnostic préchargement/Piper de la section précédente concerne le head avant
ces protections ; ses réponses étaient identiques à sa référence. Il ne faut pas
lire cette comparaison comme une promesse d’égalité textuelle après les corrections
sémantiques. La [PR #5](https://github.com/kaneda-fr/Aria/pull/5) contient le pilote et
ces protections ; une revue sur le head corrigé est requise avant de conclure.

## Deuxième revue : liste demandant un état

Copilot (`5851271634`, head `2f9c066`) a signalé un défaut valide :
le cadrage « liste » pouvait gagner sur un prédicat allumé/éteint/statut et
renvoyer l’inventaire sans lire l’état. Six variantes FR/EN échouent avant correction.
La demande d’état est maintenant prioritaire ; une liste sans prédicat reste une
énumération sans I/O. Les noms d’équipements contenant un mot d’état (« On »,
« Allumée », etc.) ne sont pas des prédicats : quatre cas de régression
supplémentaires échouent sur la première correction, puis passent.
La proposition relative anglaise après le type ou un nom sélectionné est reconnue
sans ignorer globalement « that » dans les noms ou lieux. Trois tests gardent les
qualificatifs de lieux inconnus.

Validation finale : **113 tests ciblés, 865 tests locaux et 843 tests serveur
passent**. Les 22 cas spécifiques au client Mac sont exécutés avec mocks localement.
Le paquet contient exactement les cinq modules de lecture/statuts vérifiés, sans
configuration privée. GitNexus : impact amont du résolveur faible, appelant direct
`_process_request`, worker partagé ; changement détecté de risque moyen sur quatre flux.

Deux campagnes réelles de 40 requêtes FR/EN passent : 22 lectures aux IDs exacts,
huit commandes log-only, quatre couples clarification/reprise, quatre questions
générales et deux énumérations sans lecture. Les quatre nouvelles listes filtrées
lisent l’état, respectent le prédicat et n’utilisent pas le modèle.
La dernière campagne reprend la protection des noms : **40/40 réponses identiques**
à la précédente, aucune anomalie brute ou après audit indépendant de la langue,
de la voix, des valeurs et du filtrage. Les deux contrôles de file passent.

| Groupe comparable (7 lectures/langue) | Candidat avant garde des noms | Final |
| --- | ---: | ---: |
| FR médiane requête → premier WAV | 1,077 s | 1,092 s |
| FR maximum | 1,332 s | 1,335 s |
| EN médiane requête → premier WAV | 1,118 s | 1,140 s |
| EN maximum | 1,319 s | 1,236 s |
| File FR | 4,787 s | 4,824 s |
| File EN | 4,398 s | 4,483 s |

Par rapport à la campagne de 34 cas après les premières corrections, les médianes
du même groupe sont 0,997 → 1,092 s FR et 1,182 → 1,140 s EN. Ces passages uniques
ne prouvent pas une absence de régression statistique ; aucun gain uniforme n’est
revendiqué. Le diagnostic Piper/préchargement antérieur reste ouvert et les réglages
du modèle, de l’historique et du TTS restent identiques.

Audit final : 46 admissions/46 issues terminales, deux rejets de surcharge attendus,
42 destinations réseau d’inférence observées toutes loopback, publications du pont
info-get uniquement. Tous les services d’essai sont arrêtés après collecte ; aucun
micro, haut-parleur physique ou mouvement d’équipement. La [PR et ses rapports de
revue](https://github.com/kaneda-fr/Aria/pull/5) portent le statut de livraison courant ;
cette tâche ne déploie pas le service de production.

Suite à la troisième revue, voir la [correction C5 des noms de volets](status-cover-name-review-2026-11-11.md)
pour la validation du build suivant ; les 40 cas ci-dessus précèdent cette correction.

Suite : [quatrième revue et validation partagée C6](status-query-roles-2026-11-11.md).
