"""Live-test providers: cached inventory and logged intent, never MQTT publication."""
import hashlib
import shutil
import socket

from app.plugins.actions.provider import ActionsProvider
from app.plugins.inventory.provider import InventoryProvider


class LogOnlyActions(ActionsProvider):
    execution_mode = 'log_only'

    def start(self):
        self._mark_started()

    def stop(self):
        self._mark_stopped()

    def execute_command(self, cmd_id, value=None, execution_spec=None):
        inventory = self.config['inventory']
        command = inventory.indexer.get_command(cmd_id)
        error = None
        payload = None
        if command is None:
            error = 'unknown_command'
        else:
            execution_spec = command.execution or {}
            error = self._validate_parameters(cmd_id, value, execution_spec)
            if error is None:
                payload = self._build_payload(cmd_id, value, execution_spec)
        result = {'success': error is None, 'cmd_id': cmd_id,
                  'dry_run': True, 'executed': False, 'error': error,
                  'message': ('LOG ONLY: command recorded; no device was changed.'
                              if error is None else 'LOG ONLY: command rejected; no device was changed.')}
        self.config['log']('Live.Command', request_id=self.config['request_id'](),
                           cmd_id=cmd_id, value=value, payload=payload,
                           topic=f'{self.root_topic}/cmd/set/{cmd_id}', result=result)
        return result


def install_live_tools(app, run, log, request_id, source):
    if {name for _, name in socket.if_nameindex()} != {'lo'}:
        raise RuntimeError('Live log-only tools require the isolated network namespace')
    registry = app.state.plugin_registry
    if registry.plugin_count:
        raise RuntimeError('Unexpected providers already enabled')
    cache = run / 'inventory-snapshot.json'
    shutil.copyfile(source, cache)
    import os
    import json
    from app.plugins.inventory.state_read import ReadOnlyInventory
    read_socket = os.environ.get('ARIA_JEEDOM_READ_SOCKET')
    registry.register_plugin_class('inventory', ReadOnlyInventory if read_socket else InventoryProvider)
    inventory = registry.create_plugin('inventory', {'cache_path': str(cache), 'sources': {}, **({'read_socket': read_socket} if read_socket else {})})
    inventory.start()
    if not inventory.indexer.command_count:
        raise RuntimeError('Cached inventory has no commands')
    registry.register_plugin_class('actions', LogOnlyActions)
    actions = registry.create_plugin('actions', {
        'inventory': inventory, 'log': log, 'request_id': request_id})
    actions.start()
    log('Live.ToolsReady', mode='log_only', devices=inventory.indexer.device_count,
        commands=inventory.indexer.command_count, inventory_source='live_read_bridge' if read_socket else 'cached_snapshot',
        bridge=getattr(inventory, 'bridge_status', None),
        inventory_seed_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        inventory_sha256=hashlib.sha256(json.dumps(
            inventory.get_snapshot()['devices'], sort_keys=True).encode()).hexdigest())
