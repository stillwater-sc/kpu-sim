#!/bin/bash
# scripts/release.sh
# Release tagging script for KPU Simulator
#
# Usage:
#   ./scripts/release.sh 0.3.0           # Tag a release
#   ./scripts/release.sh 0.3.0 --push    # Tag and push to origin
#   ./scripts/release.sh --current       # Show current version
#
# The version is derived from git tags. This script creates annotated tags.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_current_version() {
    if git describe --tags --match "v[0-9]*" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}Release version:${NC} $(git describe --tags --match 'v[0-9]*' --abbrev=0 2>/dev/null || echo 'none')"
        echo -e "${GREEN}Full version:${NC} $(git describe --tags --match 'v[0-9]*' 2>/dev/null || echo 'none')"
    else
        echo -e "${YELLOW}No version tags found${NC}"
    fi
}

validate_version() {
    local version=$1
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo -e "${RED}Error: Invalid version format '$version'${NC}"
        echo "Expected format: MAJOR.MINOR.PATCH (e.g., 0.3.0)"
        exit 1
    fi
}

check_clean_tree() {
    if ! git diff-index --quiet HEAD --; then
        echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

tag_exists() {
    git tag -l "v$1" | grep -q "v$1"
}

create_tag() {
    local version=$1
    local tag="v$version"

    echo -e "${GREEN}Creating tag: $tag${NC}"

    # Create annotated tag
    git tag -a "$tag" -m "Release $version

KPU Simulator version $version

See CHANGELOG.md for details."

    echo -e "${GREEN}Tag created successfully: $tag${NC}"
}

push_tag() {
    local version=$1
    local tag="v$version"

    echo -e "${GREEN}Pushing tag to origin...${NC}"
    git push origin "$tag"
    echo -e "${GREEN}Tag pushed: $tag${NC}"
}

show_usage() {
    echo "Usage: $0 VERSION [--push]"
    echo "       $0 --current"
    echo ""
    echo "Examples:"
    echo "  $0 0.3.0          Create tag v0.3.0"
    echo "  $0 0.3.0 --push   Create and push tag v0.3.0"
    echo "  $0 --current      Show current version"
    echo ""
    echo "Semantic Versioning:"
    echo "  MAJOR.MINOR.PATCH"
    echo "  - MAJOR: Breaking changes"
    echo "  - MINOR: New features, backwards compatible"
    echo "  - PATCH: Bug fixes, backwards compatible"
}

# Main script
if [[ $# -eq 0 ]]; then
    show_usage
    exit 0
fi

case "$1" in
    --current|-c)
        show_current_version
        exit 0
        ;;
    --help|-h)
        show_usage
        exit 0
        ;;
    *)
        VERSION=$1
        PUSH=false
        if [[ "$2" == "--push" ]]; then
            PUSH=true
        fi
        ;;
esac

# Validate version format
validate_version "$VERSION"

# Check for uncommitted changes
check_clean_tree

# Check if tag already exists
if tag_exists "$VERSION"; then
    echo -e "${RED}Error: Tag v$VERSION already exists${NC}"
    echo "Use 'git tag -d v$VERSION' to delete it first, or choose a different version"
    exit 1
fi

# Show what we're about to do
echo ""
echo -e "${GREEN}=== KPU Simulator Release ===${NC}"
echo "Version: $VERSION"
echo "Tag: v$VERSION"
echo ""

read -p "Create this tag? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 1
fi

# Create the tag
create_tag "$VERSION"

# Push if requested
if $PUSH; then
    push_tag "$VERSION"
fi

echo ""
echo -e "${GREEN}=== Release Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Push the tag:  git push origin v$VERSION"
echo "  2. Create GitHub release from the tag"
echo "  3. Update CHANGELOG.md if not done"
echo ""
show_current_version
