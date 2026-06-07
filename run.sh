#!/usr/bin/env bash
# Lança o petBionic CSV Analyzer
#
# NOTA: o ambiente virtual (.venv) NÃO pode viver dentro de ~/Desktop/petBionic
# porque essa pasta está sincronizada com o iCloud Drive. O iCloud despeja o
# conteúdo dos .dylib para a nuvem ("dataless") e o Qt deixa de conseguir
# carregar os plugins (erro "Could not find the Qt platform plugin cocoa").
# Por isso criamos o venv em ~/Library/Caches, que o iCloud não sincroniza.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/Library/Caches/petbionic-analysis/venv"
PYTHON="$(brew --prefix)/bin/python3"

if [ ! -x "$VENV/bin/python" ]; then
  echo "A criar ambiente virtual (fora do iCloud) e a instalar dependências..."
  rm -rf "$VENV"
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"
fi

# Descobre o caminho dos plugins Qt a partir do PyQt6 instalado e exporta-o
# ANTES do exec (o Qt lê a variável no arranque do processo).
QT_PLUGINS="$("$VENV/bin/python" -c \
  "import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), 'Qt6', 'plugins'))" \
  2>/dev/null)"

if [ -n "$QT_PLUGINS" ] && [ -d "$QT_PLUGINS/platforms" ]; then
  export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGINS/platforms"
  export QT_PLUGIN_PATH="$QT_PLUGINS"
fi

# corre o pacote a partir de src/ (layout standard)
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# exec -a define o argv[0] → o macOS mostra "PetBionic Analyser" na barra de
# menu (em vez de "Python").
exec -a "PetBionic Analyser" "$VENV/bin/python" -m petbionic_analyser "$@"
