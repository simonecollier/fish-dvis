#!/bin/bash

# GPU Queue Script - Runs commands sequentially on a single GPU
# Supports dynamic queue: add commands to queue file while running
# Can have separate queue txt files for each GPU and run them in different terminals
# Usage:
#   ./gpu_queue.sh /path/to/my_queue.txt
#


# Don't use set -e here - we handle errors explicitly
# set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default queue file
QUEUE_FILE="${QUEUE_FILE:-gpu_queue_commands.txt}"
LOG_FILE="${LOG_FILE:-gpu_queue.log}"
USE_DYNAMIC_QUEUE=false

# Function to log messages
log() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${message}"
    # Write to log file (ignore errors to prevent script from exiting)
    echo "[${timestamp}] ${message}" >> "$LOG_FILE" 2>/dev/null || true
}

# Function to run a single command
run_command() {
    local cmd="$1"
    local cmd_num="$2"
    local total="$3"
    
    log "${BLUE}========================================${NC}"
    log "${BLUE}[${cmd_num}/${total}] Starting command:${NC}"
    log "${YELLOW}${cmd}${NC}"
    log "${BLUE}========================================${NC}"
    
    local start_time=$(date +%s)
    
    # Run the command
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        log "${GREEN}✓ Command ${cmd_num}/${total} completed successfully${NC}"
        log "${GREEN}  Duration: ${hours}h ${minutes}m ${seconds}s${NC}"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log "${RED}✗ Command ${cmd_num}/${total} failed${NC}"
        log "${RED}  Duration: ${duration}s${NC}"
        return 1
    fi
}

# Function to read commands from queue file
read_queue_file() {
    local queue_file="$1"
    local processed_lines="${2:-0}"
    local new_commands=()
    
    if [ ! -f "$queue_file" ]; then
        return 0
    fi
    
    local line_num=0
    while IFS= read -r line || [ -n "$line" ]; do
        ((line_num++))
        if [ $line_num -le $processed_lines ]; then
            continue
        fi
        
        # Skip empty lines and comments
        if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
            new_commands+=("$line")
        fi
    done < "$queue_file"
    
    # Return commands via global array (bash limitation)
    QUEUE_NEW_COMMANDS=("${new_commands[@]}")
    QUEUE_PROCESSED_LINES=$line_num
}

# Main execution
main() {
    local commands=()
    local queue_processed_lines=0
    local file_path_provided=false
    
    # Check if first argument is a file path (simple heuristic: doesn't start with --)
    if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
        # Check if it's a file or looks like a file path
        # Treat as file if: exists as file, has common text file extension, or contains path separator
        if [ -f "$1" ] 2>/dev/null || [[ "$1" =~ \.(txt|queue|list|cmds)$ ]] || [[ "$1" == */* ]]; then
            # First argument looks like a file path
            QUEUE_FILE="$1"
            USE_DYNAMIC_QUEUE=true
            file_path_provided=true
            shift
        fi
    fi
    
    # Parse remaining arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --queue-file)
                QUEUE_FILE="$2"
                USE_DYNAMIC_QUEUE=true
                file_path_provided=true
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [QUEUE_FILE] [OPTIONS]"
                echo "   OR: $0 [OPTIONS] [command1] [command2] ..."
                echo ""
                echo "Arguments:"
                echo "  QUEUE_FILE           Path to queue file (enables dynamic queue mode)"
                echo "                       Commands can be added to this file while running"
                echo ""
                echo "Options:"
                echo "  --queue-file FILE    Use dynamic queue file (alternative to positional arg)"
                echo "  --help, -h           Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 my_queue.txt                    # Use queue file (recommended)"
                echo "  $0 /path/to/queue.txt              # Use queue file with full path"
                echo "  $0 \"cmd1\" \"cmd2\"                 # Pass commands directly"
                echo "  $0 --queue-file my_queue.txt       # Use queue file (alternative)"
                echo "  $0 < commands.txt                  # Read from stdin"
                echo ""
                echo "To add commands while running (with queue file):"
                echo "  echo 'new_command' >> my_queue.txt"
                echo "  # Or edit directly: nano my_queue.txt"
                exit 0
                ;;
            *)
                commands+=("$1")
                shift
                ;;
        esac
    done
    
    # If using dynamic queue (file path provided), read from file
    if [ "$USE_DYNAMIC_QUEUE" = true ] && [ "$file_path_provided" = true ]; then
        # Get absolute path of queue file
        if [[ "$QUEUE_FILE" != /* ]]; then
            # Relative path - make it absolute
            QUEUE_FILE="$(cd "$(dirname "$QUEUE_FILE")" 2>/dev/null && pwd)/$(basename "$QUEUE_FILE")" || QUEUE_FILE="$(pwd)/$QUEUE_FILE"
        fi
        
        # Check if file exists
        if [ ! -f "$QUEUE_FILE" ]; then
            log "${YELLOW}Queue file does not exist. Creating: ${QUEUE_FILE}${NC}"
            touch "$QUEUE_FILE"
        fi
        
        # Read initial commands from queue file
        read_queue_file "$QUEUE_FILE" 0
        commands=("${QUEUE_NEW_COMMANDS[@]}")
        queue_processed_lines=$QUEUE_PROCESSED_LINES
        
        log "${CYAN}Using queue file: ${QUEUE_FILE}${NC}"
        log "${CYAN}Add commands with: echo 'command' >> ${QUEUE_FILE}${NC}"
        log "${CYAN}Or edit directly with: nano ${QUEUE_FILE} (or vim, emacs, etc.)${NC}"
    elif [ "$USE_DYNAMIC_QUEUE" = true ] && [ "$file_path_provided" = false ]; then
        # Using --queue-file option, initialize queue file
        # Get absolute path of queue file
        if [[ "$QUEUE_FILE" != /* ]]; then
            # Relative path - make it absolute
            QUEUE_FILE="$(cd "$(dirname "$QUEUE_FILE")" 2>/dev/null && pwd)/$(basename "$QUEUE_FILE")" || QUEUE_FILE="$(pwd)/$QUEUE_FILE"
        fi
        
        # Create queue file with initial commands (or create empty file)
        if [ ${#commands[@]} -gt 0 ]; then
            > "$QUEUE_FILE"  # Clear existing file
            for cmd in "${commands[@]}"; do
                echo "$cmd" >> "$QUEUE_FILE"
            done
        else
            # Create empty file if it doesn't exist
            touch "$QUEUE_FILE"
        fi
        
        log "${CYAN}Dynamic queue enabled. Queue file: ${QUEUE_FILE}${NC}"
        log "${CYAN}Add commands with: echo 'command' >> ${QUEUE_FILE}${NC}"
        log "${CYAN}Or edit directly with: nano ${QUEUE_FILE} (or vim, emacs, etc.)${NC}"
    elif [ ${#commands[@]} -eq 0 ]; then
        # No commands and no queue file - read from stdin
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments
            if [[ -n "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
                commands+=("$line")
            fi
        done
    fi
    
    # Check if we have any commands
    # If using dynamic queue, allow empty commands (will monitor file for new commands)
    if [ ${#commands[@]} -eq 0 ] && [ "$USE_DYNAMIC_QUEUE" != true ]; then
        echo "Error: No commands provided"
        echo "Usage: $0 [QUEUE_FILE]"
        echo "   OR: $0 \"command1\" \"command2\" ..."
        echo "   OR: $0 < commands.txt"
        echo "   OR: $0 --queue-file queue.txt"
        exit 1
    fi
    
    # If using dynamic queue but no commands found, warn but continue
    if [ ${#commands[@]} -eq 0 ] && [ "$USE_DYNAMIC_QUEUE" = true ]; then
        log "${YELLOW}No commands found in queue file. Monitoring for new commands...${NC}"
    fi
    
    local total=${#commands[@]}
    if [ "$USE_DYNAMIC_QUEUE" = true ]; then
        if [ $total -gt 0 ]; then
            log "${BLUE}GPU Queue: ${total} initial command(s), monitoring ${QUEUE_FILE} for more${NC}"
        else
            log "${BLUE}GPU Queue: Monitoring ${QUEUE_FILE} for commands${NC}"
        fi
    else
        log "${BLUE}GPU Queue: ${total} command(s) queued${NC}"
    fi
    log "${BLUE}Log file: ${LOG_FILE}${NC}"
    echo ""
    
    local success_count=0
    local fail_count=0
    local cmd_num=0
    local all_commands=("${commands[@]}")
    
    # Main processing loop
    while true; do
        # If using dynamic queue, check for new commands only after previous command completed
        if [ "$USE_DYNAMIC_QUEUE" = true ]; then
            read_queue_file "$QUEUE_FILE" "$queue_processed_lines"
            if [ ${#QUEUE_NEW_COMMANDS[@]} -gt 0 ]; then
                for new_cmd in "${QUEUE_NEW_COMMANDS[@]}"; do
                    all_commands+=("$new_cmd")
                done
                queue_processed_lines=$QUEUE_PROCESSED_LINES
            fi
        fi
        
        # Process commands from array
        if [ ${#all_commands[@]} -gt 0 ]; then
            local cmd="${all_commands[0]}"
            all_commands=("${all_commands[@]:1}")  # Remove first element
            ((cmd_num++))
            
            local total_display="?"
            if [ "$USE_DYNAMIC_QUEUE" != true ]; then
                total_display="$total"
            fi
            
            if run_command "$cmd" "$cmd_num" "$total_display"; then
                ((success_count++))
            else
                ((fail_count++))
                log "${RED}Stopping queue due to error${NC}"
                exit 1
            fi
            echo ""
        else
            # No more commands in array - exit
            # For dynamic queue, we already checked the file above, so if array is empty, there are no more commands
            break
        fi
    done
    
    # Summary
    local total_processed=$((success_count + fail_count))
    log "${BLUE}========================================${NC}"
    log "${BLUE}Queue completed!${NC}"
    log "${GREEN}Successful: ${success_count}/${total_processed}${NC}"
    if [ $fail_count -gt 0 ]; then
        log "${RED}Failed: ${fail_count}/${total_processed}${NC}"
    fi
    log "${BLUE}========================================${NC}"
    
    # Clean up queue file if using dynamic queue
    if [ "$USE_DYNAMIC_QUEUE" = true ] && [ -f "$QUEUE_FILE" ]; then
        log "${CYAN}Queue file ${QUEUE_FILE} preserved. Delete manually if no longer needed.${NC}"
    fi
}

# Run main function
main "$@"

