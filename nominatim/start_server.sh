#!/bin/bash

# Set default PBF URL if not provided
if [ -z "${PBF_URL}" ]; then
  # PBF_URL=https://download.geofabrik.de/europe-latest.osm.pbf
  # PBF_URL=https://download.geofabrik.de/central-europe-latest.osm.pbf
  PBF_URL=https://download.geofabrik.de/europe/hungary-latest.osm.pbf
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_success() {
    echo -e "${GREEN}SUCCESS: $1${NC}"
}

print_info() {
    echo -e "$1"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running or not accessible."
        print_info "Please start Docker service or check Docker permissions."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if port 8080 is available
check_port() {
    local port=8080
    if command -v netstat &> /dev/null; then
        if netstat -tuln | grep -q ":${port} "; then
            print_error "Port ${port} is already in use"
            print_info "Please stop the service using port ${port} or modify the script to use a different port"
            exit 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -tuln | grep -q ":${port} "; then
            print_error "Port ${port} is already in use"
            print_info "Please stop the service using port ${port} or modify the script to use a different port"
            exit 1
        fi
    else
        print_warning "Cannot check if port ${port} is available (netstat/ss not found)"
    fi
    
    print_success "Port ${port} appears to be available"
}

# Check available disk space (Nominatim can require significant space)
check_disk_space() {
    local required_gb=10  # Minimum required space in GB
    local available_kb=$(df . | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space. Available: ${available_gb}GB, Required: ${required_gb}GB"
        print_info "Nominatim requires significant disk space for map data processing"
        exit 1
    fi
    
    print_success "Sufficient disk space available (${available_gb}GB)"
}

# Check if PBF URL is accessible
check_pbf_url() {
    local url="$1"
    print_info "Checking if PBF URL is accessible: $url"
    
    if command -v curl &> /dev/null; then
        if ! curl --head --silent --fail "$url" &> /dev/null; then
            print_error "Cannot access PBF URL: $url"
            print_info "Please check your internet connection and verify the URL is correct"
            exit 1
        fi
    elif command -v wget &> /dev/null; then
        if ! wget --spider --quiet "$url" &> /dev/null; then
            print_error "Cannot access PBF URL: $url"
            print_info "Please check your internet connection and verify the URL is correct"
            exit 1
        fi
    else
        print_warning "Cannot verify PBF URL accessibility (curl/wget not found)"
        print_info "Proceeding without URL verification..."
        return 0
    fi
    
    print_success "PBF URL is accessible"
}

# Check if Docker image exists locally or can be pulled
check_docker_image() {
    local image="mediagis/nominatim:5.1"
    
    print_info "Checking Docker image: $image"
    
    if docker image inspect "$image" &> /dev/null; then
        print_success "Docker image '$image' found locally"
        return 0
    fi
    
    print_info "Image not found locally, attempting to pull..."
    if ! docker pull "$image"; then
        print_error "Failed to pull Docker image: $image"
        print_info "Please check your internet connection and Docker Hub accessibility"
        exit 1
    fi
    
    print_success "Docker image '$image' pulled successfully"
}

# Main prerequisite checks
print_info "=== Nominatim Server Startup - Prerequisite Checks ==="

check_docker
check_port
check_disk_space
check_docker_image
check_pbf_url "$PBF_URL"

print_success "All prerequisite checks passed!"
print_info "=== Starting Nominatim Server ==="

# Add error handling for docker run
if ! docker run -it -e PBF_URL="$PBF_URL" -p 8080:8080 mediagis/nominatim:5.1; then
    print_error "Failed to start Nominatim container"
    exit 1
fi
