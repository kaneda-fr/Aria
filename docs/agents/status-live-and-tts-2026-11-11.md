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
