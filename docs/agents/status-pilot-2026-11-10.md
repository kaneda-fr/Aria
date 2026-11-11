# Pilote de statuts lumières et volets — 2026-11-10

Implémentation dans le worktree attaché `codex/lights-questions`, sur main `7ba8250`
avec le correctif lumières hérité. Ces essais précèdent la préparation de PR du
[suivi](status-live-and-tts-2026-11-11.md). Le checkout canonique n’est pas la
source de cette implémentation. [Étude](jeedom-status-semantics-2026-10-21.md) et
[contrat](jeedom-status-contract-2026-10-21.md) décrivent aussi des extensions hors pilote.
Les essais utilisent uniquement le serveur, du texte, du PCM sauvegardé et des WAV
capturés ; aucun microphone, haut-parleur ou appareil réel n’est actionné.

## Comportement livré

- Une projection partagée produit des faits typés, les valeurs sources, la couverture,
  les limites et la provenance. Les résultats d’outils destinés au LLM contiennent les
  faits ciblés ; les observations techniques restent dans les journaux.
- Les lectures simples restent sans génération de réponse par un modèle. Le travail
  spéculatif de cache d’historique existant reste séparé et n’est pas supprimé.
- Parole FR/EN quotidienne : « une partie des lumières est allumée », « le volet est
  à moitié ouvert », « je ne peux pas déterminer sa position ». Aucun canal/circuit
  technique à l’oral ; les noms d’équipements restent ceux de l’inventaire.
- Lumières : état mixte et inconnu distincts, luminosité distincte de la position d’un
  volet ; une unité incompatible bloque l’interprétation. Pas de déduction depuis W.
- Volets : lecture individuelle ou groupée, filtres de pièce, choix après clarification,
  et relances adjacentes du même locuteur. Les équipements sans retour d’état restent
  dans la portée ; ils ne deviennent pas implicitement fermés.
- Une affirmation « tous » exige une couverture interprétable de toute la portée.
  Un témoin positif peut répondre à « au moins un ». Un catalogue incomplet n’autorise
  pas à généraliser à toute la maison. L’âge inconnu reste signalé en mots simples.
- Si les propositions d’une clarification changent, une réponse ordinale repose la
  question. Une modification de définition pendant une lecture invalide les anciennes
  interprétations. La sélection, les limites de huit infos, l’autorisation d’action,
  la file commune et les trois échanges d’historique sont conservés.

Le pilote n’est pas un nouvel interpréteur universel : périodes, diagnostics, météo,
capteurs et compteurs conservent leur traitement actuel. Des formulations nouvelles,
la parole réelle et le sens des données de chaque installation restent à vérifier.

## Convention des volets réels

Ne pas supposer que 0 signifie fermé ou que 100 signifie ouvert. Le pilote exige
un fichier local `ARIA_STATUS_MAP_FILE`, schéma `aria.status-map/1`, avec les bornes
`closed_value` / `open_value` et l’identité exacte de chaque propriété. Les bornes
peuvent être inversées ou utiliser une autre échelle documentée, par exemple 0–99.
Sans convention vérifiée, la position est inconnue ; aucune calibration par mouvement
n’a été effectuée. Les conventions de la campagne initiale sont **synthétiques**, pas une validation
physique des volets de l’installation.

Les mappings sont liés à source, équipement, nom, pièce, commande, sous-type,
capacité, unité et plage. Les changements exigent une revalidation. Le fichier est
lu localement et mis en cache par chemin : redémarrer le serveur après modification.
Une erreur de fichier n’entraîne aucun téléchargement ni interprétation par défaut.
Les dates absentes restent inconnues ; `max_age_s` ne crée pas une date de collecte.
Voir [configuration](../ENVIRONMENT.md#typed-lightcover-status-pilot).

## Validation et limites des mesures

**824 tests locaux et 802 tests serveur passent** (Python 3.12 / 3.11, quatre
avertissements de dépendances déjà connus). Les 22 tests Mac exclus du serveur ont
été exécutés localement avec des dépendances audio simulées.

La référence avant pilote et un candidat intermédiaire ont chacun exécuté 54 cas
avec synthèse capturée et deux scénarios de file. Ce candidat a révélé une formulation
anglaise non reconnue ; elle a été corrigée. La validation de transfert finale porte
sur **56 cas FR/EN** : 36 lectures avec IDs et valeurs attendus, zéro génération
foreground pour ces lectures, huit commandes exactes log-only et deux scénarios de
file sans échec. Deux pièces inexistantes ne déclenchent aucune lecture globale.
Sur les 34 lectures communes, les IDs lus sont corrects dans 14 cas en référence,
puis dans les 34 cas corrigés ; deux demandes supplémentaires portent le transfert à 36.
Les **18 réponses hors statuts** comparées restent littéralement identiques à la
référence, y compris calculs, clarification et confirmations simulées. Cela ne prouve
pas la qualité de toutes les questions générales ; les défauts scientifiques français
historiques ne sont pas corrigés par ce pilote.

La validation finale de qualité conserve cinq synthèses de textes représentatifs et
deux WAV d’entrée sauvegardés. Les 49 autres cas attendent explicitement une réponse
silencieuse et vérifient l’absence d’audio : leur temps sans WAV ne vaut jamais zéro
latence vocale. Un dernier durcissement fige la définition avant tout `await` de lecture,
même si un fournisseur la modifie en place. Le code final ainsi durci a été retesté
par la suite serveur complète et quatre requêtes FR/EN avec WAV capturé.

Seul drapeau automatique restant : la transcription anglaise écrit « Arya » au lieu
d’« Aria », comme la référence. L’alias de réveil fonctionne, la lecture attendue a
lieu et le rendu vocal utilise la langue demandée. Aucun test acoustique ni écoute
humaine sur matériel n’est revendiqué.

### Latence et ressources

Texte accepté → premier WAV disponible, sur le Xeon E3-1225 v5, quatre cœurs/threads,
Qwen2.5 3B, température 0, mêmes voix Piper locales et historique de six messages.
Les mesures comprennent la file mais pas le transport d’un microphone réel ni la
lecture physique. La comparaison complète ci-dessous précède uniquement le dernier
correctif d’adverbe anglais et le durcissement contre une mutation en place ; les
chemins lumières et le rendu normal mesurés ne changent pas avec ces correctifs.

| Mesure | Référence | Pilote complet |
| --- | --- | --- |
| Lumières FR, médiane sur cinq textes | 0,871 s | 0,999 s |
| Lumières FR, p95 empirique / maximum | 1,773 s | 1,071 s |
| Lumières EN, médiane sur cinq textes | 0,978 s | 0,986 s |
| Lumières EN, p95 empirique / maximum | 1,541 s | 2,316 s |
| File FR, requête acceptée → WAV | 4,795 s | 4,779 s |
| File EN, requête acceptée → WAV | 4,653 s | 4,409 s |

Avec cinq observations, le p95 au rang supérieur est le maximum ; ce n’est pas un
SLO ni une estimation statistique solide. **Pas de gain uniforme** : la médiane FR
augmente de 0,128 s. Le premier bilan FR s’améliore 1,773→0,999 s, mais le premier
bilan EN après la conversation FR est plus lent, 1,541→2,316 s. Sur ce dernier, la
réponse textuelle est prête en environ 24 ms ; la phase Piper/WAV prend environ
2,29 s malgré un texte plus court. Les traces ne prouvent pas que le préchargement
LLM cause ce pic. La concurrence CPU, le démarrage de voix et le cache sont à isoler.

Lors du transfert à cadence différente, les premiers WAV lumières FR/EN sont
1,046 / 1,007 s et les volets 0,843 / 0,867 s. Les deux nouveaux essais de file donnent
4,866 / 4,749 s, sans échec (un essai par langue, donc aucune preuve d’amélioration
stable). La fumée sur le dernier code donne lumières FR/EN 0,973 / 0,992 s, volets
0,883 / 0,708 s. Ces contrôles courts ne remplacent pas la mesure après conversation
longue. Les entrées PCM sauvegardées donnent fin PCM → premier WAV 0,987 s FR /
0,967 s EN, avec réponses mises en cache ; cette métrique inclut l’ASR de ces fichiers.

Projection et rendu seuls, 1 200 appels par domaine sur le serveur : p50/p95 environ
**0,050 / 0,064 ms** pour cinq états de lumières et **0,039 / 0,053 ms** pour trois
volets. Le maximum lumières est 3,54 ms, premier import compris. Ce microbench utilise
les mappings préchargés et exclut lecture réseau, LLM et TTS. Le paquet de faits hors
observations pèse environ 1,9 KiB / 1,5 KiB. Le processus Python autonome, inventaire
inclus, atteint environ 84 MiB RSS ; ce n’est pas une mesure du supplément mémoire
ajouté au serveur complet. Les instantanés des processus application + modèle sont
comparables, environ 4,8 GiB de somme RSS, avec possibles pages partagées comptées
plusieurs fois. Pic global, charge des autres services et consommation électrique
ne sont pas mesurés.

### Complément : entrée vocale des volets

Deux nouveaux fichiers de parole synthétique FR/EN traversent l’ASR, la sélection,
la lecture contrôlée et le TTS du dernier code serveur. Les trois équipements et
leurs états correspondent aux données attendues ; aucun appel de génération et
aucune commande d’action. Les voix choisies sont française et anglaise respectivement.
Fin PCM → premier WAV : **1,697 s FR / 1,741 s EN** ; après admission :
0,915 / 0,890 s. Les deux voix de synthèse d’entrée diffèrent de la voix d’Aria.
Le contrôle littéral anglais signale encore « Arya » au lieu d’« Aria » ; l’alias de
réveil fonctionne et la question est correctement traitée. Aucun essai acoustique réel.
Ces deux contrôles s’ajoutent aux 56 cas fonctionnels et aux quatre contrôles courts ;
ils ne constituent pas une nouvelle campagne statistique de latence.

Le dernier profil réutilise les **802 tests réellement exécutés et réussis** du profil
précédent, avec vérification de sources identiques. Il ne relance pas pytest. Une ligne
de console historique « Final history target-server suite passed » désigne cette
validation préalable ; cette réutilisation est explicitée dans les preuves privées.

### Sûreté de l’essai et suite conseillée

Les audits réseau des cinq profils collectés observent 39, 49, 22, 15 et 18 destinations
loopback, **aucune destination externe ni tentative DNS**. Les cycles de requête sont
cohérents : respectivement 60/60, 60/60, 62/62, 4/4 et 2/2 acceptations/issues terminales,
sans doublon. Cette preuve concerne les profils isolés, pas tous les fournisseurs
facultatifs. Toutes leurs unités sont arrêtées après collecte ; aucune capture ou
lecture audio réelle n’est relancée. L’intégrité SHA256 de 103 fichiers Python d’application et de scripts du serveur
final est vérifiée contre les fichiers envoyés ; aucun processus Aria de serveur
ou de client audio Mac n’est actif au contrôle final.

La détection GitNexus du diff cumulé depuis main indique 24 fichiers, 173 symboles,
43 processus et risque CRITICAL, avec le correctif lumières hérité. Les rendus et
résolveurs HIGH ont été annoncés avant édition. Aucun contournement d’autorisation
n’a été trouvé en relecture ; contrôles exacts et dry-run ont été vérifiés séparément.
Les modifications étaient non commitées pendant cette campagne initiale. L’étude, le contrat et le contexte courant
sont actualisés ; les autres sources héritées du checkout canonique sont préservées.

Prochaine priorité de latence : isoler le pic Piper/WAV anglais sous cadence de
conversation, sans diminuer l’historique, falsifier les états ou ajouter un modèle.
Pour interpréter toute la course des volets réels, confirmer les deux bornes et
renseigner les mappings privés ; la calibration partielle ci-dessous ne suffit pas. Aucune position physique n’a été présumée ni testée par mouvement.
La classification utilise les capacités et catégories existantes ; une catégorie
`opening` mal renseignée peut demander une correction d’inventaire pour un bilan précis.
Les données et adresses privées restent dans `.local/status-pilot/` du checkout
canonique : source de référence archivée avant changement, manifestes SHA256, données
contrôlées, sorties de tests, traces réseau et rapports par requête.

Les profils utilisent les mêmes modèles et le même inventaire mis en cache, six
messages d’historique, les mêmes observations contrôlées et les outils log-only.
Chaque profil tourne seul dans un espace réseau loopback ; le pont synthétique
communique par socket Unix et ne possède aucun client MQTT. Aucun lecteur audio
physique n’est enregistré. Le moniteur du service réel n’est pas redirigé vers ces essais.
Les latences rapportées sont celles du texte/PCM à la disponibilité d’un WAV capturé,
pas une mesure acoustique micro → haut-parleur ni une preuve de fraîcheur Jeedom.

La relecture indépendante a reproduit cinq défauts, puis des tests RED→GREEN ont
vérifié leurs corrections : portée manquante pour « tous », volets sans état exclus,
choix ordinal périmé, définition lumières modifiée pendant la lecture et unités
incompatibles. Les essais longs ont aussi corrigé la perte du domaine après réduction de l’historique,
la relance de pièce passant d’un à plusieurs volets, et les adverbes de prédicat
anglais sans ignorer une pièce nommée « Still ». Un autre test vérifie qu’une lecture ciblant un seul canal ne projette
pas les propriétés non demandées. Les assertions de vocabulaire des anciens tests
ont été adaptées ; les contrôles d’IDs, de portée et d’effets sont conservés.

Les options de clarification sont comparées par leurs libellés ordonnés effectivement
présentés. Un remplacement conservant exactement les mêmes libellés ne peut pas être
identifié par le seul texte historique ; la lecture et le mapping sont à nouveau
validés contre l’identité courante. Des libellés identiques ne permettent pas un choix
ordinal. Aucun choix de statut ne fournit une autorisation d’action.

[Suivi : calibration locale fermé uniquement](cover-calibration-2026-11-10.md) :
deux propriétés configurées, une indisponible ; validation et limites séparées.

## Reproduction sûre

Dans le worktree :

```sh
python -m pytest tests/test_status_facts.py tests/test_status_pilot_review.py tests/test_cover_status.py tests/test_light_questions.py tests/test_light_question_safety.py tests/test_read_integration.py -q
python -m pytest tests -q
python3 scripts/check_context.py --base main
```

Sur le serveur préparé, réutiliser les profils isolés et preuves privées de ce pilote,
avec l’environnement du [guide E2E](e2e-testing.md). Ne pas exécuter les scripts
historiques à la racine, dont certains publient des commandes réelles. Arrêter les
services isolés après collecte ; ne pas relancer les clients audio.
