# Argus Project Restructuring Plan

## ✅ Step 1: Create destination directories at root level
## ✅ Step 2: Move files from `core/` to root (using PowerShell)
## ✅ Step 3: Fix `backend/services/__init__.py` imports
## ✅ Step 4: Fix `backend/api/main.py` imports
## ✅ Step 5: Fix `backend/api/stream_routes.py` imports
## ✅ Step 6: Fix `backend/api/stream_ws.py` imports
## ✅ Step 7: Fix `backend/scripts/*.py` imports
## ✅ Step 8: Fix `backend/config/config.py` path resolution
## ✅ Step 9: Fix `backend/database/db.py` path resolution
## ✅ Step 10: Fix Dockerfiles (paths)
## ✅ Step 11: Fix docker-compose files (volume mounts, context)
## ✅ Step 12: Fix run_app.bat and run_webcam.bat
## ✅ Step 13: Fix Django admin settings.py paths
## ✅ Step 14: Fix internal service imports (processing_coordinator, rules_engine, stream_ingestion, video_pipeline)
## ✅ Step 15: Update README.md structure diagram
## ✅ Step 16: Remove empty `core/` directory
## ✅ Step 17: Clean up and verify - remove core/ if still exists, verify all files

---

## ✅ Recent Import Fixes (Completed)

### Fixed Broken Relative Imports
| # | File | Original Import | Fixed Import |
|---|------|----------------|--------------|
| ✅ | `backend/services/management/telemetry_monitor.py` | `from config.config import get_config` | `from ...config.config import get_config` (with fallback) |
| ✅ | `backend/services/management/state_recovery_manager.py` | `from config.config import get_config` | `from ...config.config import get_config` |
| ✅ | `backend/services/management/user_attention_tracker.py` | `from config.config import get_config` | `from ...config.config import get_config` |
| ✅ | `backend/services/core_engine/evolutionary_engine.py` | `from config.config import get_config` + `from management.event_store import get_event_store` | `from ...config.config import get_config` + `from ..management.event_store import get_event_store` |
| ✅ | `backend/services/core_engine/logic_mutator.py` | `from config.config import get_config` | `from ...config.config import get_config` |

