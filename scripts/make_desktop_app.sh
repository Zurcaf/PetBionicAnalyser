#!/usr/bin/env bash
# Cria/atualiza a app clicável do PetBionic Analyser.
#
# A app é um wrapper fino: ao clicar, corre run_analysis.sh, que executa o
# código-fonte atual (csv_analyzer.py). Não há "build" congelado — qualquer
# alteração ao código fica refletida no próximo clique.
#
# Detalhes importantes:
#  * O executável do bundle é um BINÁRIO COMPILADO + assinatura ad-hoc → dá
#    identidade de código estável, necessária para o macOS PEDIR acesso às pastas
#    (TestData no Desktop) e para o ícone aparecer.
#  * A app REAL vive aqui (~/Developer, fora do iCloud) para a assinatura ficar
#    limpa. No Desktop fica apenas um ATALHO (o Desktop é sincronizado pelo
#    iCloud, que poria xattrs a invalidar a assinatura).
#
# Correr:  bash "make_desktop_app.sh"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SH="$REPO_ROOT/run.sh"
ICON_SRC="$REPO_ROOT/assets/app_icon.png"
APP="$REPO_ROOT/PetBionic Analyser.app"            # app real (fora do iCloud)
LINK="$HOME/Desktop/PetBionic Analyser.app"       # atalho no Desktop
EXE_DIR="$APP/Contents/MacOS"
RES_DIR="$APP/Contents/Resources"
LAUNCH_SH="$RES_DIR/launch.sh"                     # lógica em Resources (não MacOS)
EXE="$EXE_DIR/PetBionicAnalyser"
LOG="$HOME/Library/Logs/PetBionic-Analyser.log"

if [ ! -f "$RUN_SH" ]; then
  echo "ERRO: não encontrei $RUN_SH" >&2
  exit 1
fi

rm -rf "$APP"
mkdir -p "$EXE_DIR" "$RES_DIR"

# --- ícone (.icns a partir do logótipo petBionic) --------------------------
ICON_PLIST=""
if [ -f "$ICON_SRC" ]; then
  ICONSET="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for sz in 16 32 128 256 512; do
    sips -z $sz $sz             "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}.png"     >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$ICON_SRC" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
  done
  if iconutil -c icns "$ICONSET" -o "$RES_DIR/AppIcon.icns" 2>/dev/null; then
    ICON_PLIST="  <key>CFBundleIconFile</key>        <string>AppIcon</string>"
    echo "✓ Ícone gerado a partir de $(basename "$ICON_SRC")"
  else
    echo "⚠  iconutil falhou — app fica com ícone genérico"
  fi
  rm -rf "$(dirname "$ICONSET")"
else
  echo "⚠  ícone não encontrado em $ICON_SRC — app fica com ícone genérico"
fi

# --- lógica de arranque (script bash, em Resources) -------------------------
cat > "$LAUNCH_SH" <<EOF
#!/bin/bash
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
mkdir -p "\$(dirname "$LOG")"
exec bash "$RUN_SH" >> "$LOG" 2>&1
EOF
chmod +x "$LAUNCH_SH"

# --- executável compilado (única coisa em MacOS/) ---------------------------
CSRC="$(mktemp -d)/launcher.c"
cat > "$CSRC" <<EOF
#include <unistd.h>
int main(void) {
    execl("/bin/bash", "bash", "$LAUNCH_SH", (char *)0);
    return 1;
}
EOF
cc -O2 -o "$EXE" "$CSRC"
rm -rf "$(dirname "$CSRC")"
echo "✓ Executável compilado"

# --- Info.plist -------------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>PetBionic Analyser</string>
  <key>CFBundleDisplayName</key>     <string>PetBionic Analyser</string>
  <key>CFBundleExecutable</key>      <string>PetBionicAnalyser</string>
  <key>CFBundleIdentifier</key>      <string>com.petbionic.analyser</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>CFBundleVersion</key>         <string>1.0</string>
  <key>CFBundleShortVersionString</key> <string>1.0</string>
  <key>CFBundleInfoDictionaryVersion</key> <string>6.0</string>
$ICON_PLIST
  <key>NSHighResolutionCapable</key> <true/>
  <key>LSMinimumSystemVersion</key>  <string>11.0</string>
</dict>
</plist>
EOF

# --- assinatura ad-hoc (identidade estável p/ TCC e ícone) ------------------
xattr -cr "$APP" 2>/dev/null || true
if codesign --force --sign - "$APP" 2>/dev/null && codesign --verify --strict "$APP" 2>/dev/null; then
  echo "✓ App assinada (ad-hoc) e verificada"
else
  echo "⚠  codesign falhou (FDA pode ficar menos fiável)"
fi

# --- atalho no Desktop ------------------------------------------------------
rm -f "$LINK"
ln -s "$APP" "$LINK"
echo "✓ Atalho criado no Desktop"

# --- registar na LaunchServices + refrescar ícone do Finder -----------------
LSREG="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
[ -x "$LSREG" ] && "$LSREG" -f "$APP" 2>/dev/null || true

echo "✓ App: $APP"
echo "  Atalho clicável: $LINK"
echo "  Log de arranque: $LOG"
