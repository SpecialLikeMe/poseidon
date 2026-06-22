# 1. Grab the current user path safely
# (In POSIX, PATH is already available as a variable)
OLD_PATH="$PATH"

# 2. Normalize your installation directory (resolves slashes and links)
# This gets the absolute path of the directory containing this script
INSTALL_DIR=$(cd "$(dirname "$0")" && pwd)

# 3. Check against existing paths to prevent duplicate or casing bugs
# Loop through the PATH using ':' as the delimiter
MATCH_FOUND=0
SAVED_IFS="$IFS"
IFS=":"

for p in $OLD_PATH; do
  # Clear trailing slashes for clean normalization
  CLEANED_P=$(echo "$p" | sed 's:/*$::')
  if [ "$CLEANED_P" = "$INSTALL_DIR" ]; then
    MATCH_FOUND=1
    break
  fi
done
IFS="$SAVED_IFS"

# 4. Write back to user environment safely if not found
if [ $MATCH_FOUND -eq 0 ]; then
  # Target standard POSIX user profile
  PROFILE_FILE="$HOME/.profile"

  # Append the export command to the file safely
  echo "" >>"$PROFILE_FILE"
  echo "# Added by installation script" >>"$PROFILE_FILE"
  echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >>"$PROFILE_FILE"

  # Print green-like output text manually using ANSI codes
  printf "\033[0;32mSuccessfully added to PATH without corruption or duplicates!\033[0m\n"
else
  # Print yellow-like output text manually using ANSI codes
  printf "\033[0;33mApp directory is already uniquely identified in the PATH.\033[0m\n"
fi

# ==============================================================================
# NEW STEP 5: Create the "pos" command shortcut executable in the install directory
# ==============================================================================
SHORTCUT_FILE="$INSTALL_DIR/pos"

# Write the shell script wrapper content safely
cat <<'EOF' >"$SHORTCUT_FILE"
#!/bin/sh
# Get the absolute path of the folder containing this shortcut
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
python3 "$SCRIPT_DIR/main.py" "$@"
EOF

# Make the file executable (analogous to creating a runnable .bat file)
chmod +x "$SHORTCUT_FILE"

printf "\033[0;32mSuccessfully created 'pos' shortcut command!\033[0m\n"
