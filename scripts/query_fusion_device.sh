# Sourced by query-fusion launchers; consumes only --device arguments.
# Explicit --device/DEVICE uses visible CUDA ordinals without imposing a GPU=0
# mask. Preserve caller/scheduler CUDA_VISIBLE_DEVICES; legacy GPU still works.
query_fusion_positionals=()
query_fusion_positional_count=0
query_fusion_explicit_device=0
while (( $# )); do
  case "$1" in
    --device)
      if (( $# < 2 )) || [[ -z "$2" ]]; then
        echo '--device requires cpu, cuda or cuda:N' >&2
        exit 2
      fi
      DEVICE="$2"
      query_fusion_explicit_device=1
      shift 2
      ;;
    --device=*)
      DEVICE="${1#--device=}"
      query_fusion_explicit_device=1
      shift
      ;;
    *)
      query_fusion_positionals+=("$1")
      query_fusion_positional_count=$((query_fusion_positional_count + 1))
      shift
      ;;
  esac
done
if (( query_fusion_positional_count )); then
  set -- "${query_fusion_positionals[@]}"
fi
if (( query_fusion_explicit_device )) && [[ -z "$DEVICE" ]]; then
  echo '--device requires cpu, cuda or cuda:N' >&2
  exit 2
fi
if [[ ! "${DEVICE:-cuda:0}" =~ ^cuda(:[0-9]+)?$ && "${DEVICE:-cuda:0}" != cpu ]]; then
  echo '--device must be cpu, cuda or cuda:N (for example cuda:1)' >&2
  exit 2
fi
if (( ! query_fusion_explicit_device )); then
  if [[ -n "${GPU:-}" || -z "${DEVICE:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
  fi
fi
export DEVICE="${DEVICE:-cuda:0}"
