#!/bin/bash

# Target directory for Termux URL opener
TERMUX_BIN="$HOME/bin"
OPENER="$TERMUX_BIN/termux-url-opener"

# Ensure bin directory exists
mkdir -p "$TERMUX_BIN"

# Create the opener script
cat << 'EOF' > "$OPENER"
#!/bin/bash

# Project Path (Update this if you move the project)
PROJECT_DIR="$HOME/podcast_ads"
PYTHON_SCRIPT="$PROJECT_DIR/run.py"

# Check for URL/File argument
if [ $# -eq 0 ]; then
    echo "No input provided."
    read -p "Press enter to exit"
    exit 1
fi

# Function to run the tool
run_tool() {
    MODE="$1"
    shift # Remove the mode argument, leaving only the file/url arguments in $@
    
    echo "Running in mode: $MODE"
    cd "$PROJECT_DIR"
    
    # Use uv to run (handles venv automatically)
    if [ "$MODE" == "process" ]; then
        uv run run.py "$@" --save-subs --save-transcript
    elif [ "$MODE" == "play" ]; then
        uv run run.py "$@" --play
    elif [ "$MODE" == "audio" ]; then
        uv run run.py "$@" --play-audio
    elif [ "$MODE" == "download" ]; then
        uv run run.py "$@" --save-clean-audio --save-subs
    fi
}

# Dialog Menu
echo "=================================="
echo "   Podcast Ad Skipper (Android)   "
echo "=================================="
echo "Inputs:"
for arg in "$@"; do
    echo " - $arg"
done
echo ""
echo "1. Play (Video + Skips)"
echo "2. Play (Audio Only + Skips)"
echo "3. Process Only (Save Metadata)"
echo "4. Download Clean Audio (MP3)"
echo "x. Exit"
echo ""
read -p "Select option: " CHOICE

case "$CHOICE" in
    1) run_tool "play" "$@" ;;
    2) run_tool "audio" "$@" ;;
    3) run_tool "process" "$@" ;;
    4) run_tool "download" "$@" ;;
    x) exit 0 ;;
    *) echo "Invalid option";;
esac

echo ""
read -p "Done. Press enter to close."
EOF

# Make executable
chmod +x "$OPENER"

echo "Android Setup Complete."
echo "1. Created $OPENER"
echo "2. When you 'Share' a URL/File to Termux, the menu will appear."
echo "3. Ensure you have run 'termux-setup-storage' to allow MPV integration."
