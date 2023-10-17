install:
	pip3 -v install -e . && cp build/*/compile_commands.json build/

uninstall:
	pip3 -v uninstall voxel_hash_map


