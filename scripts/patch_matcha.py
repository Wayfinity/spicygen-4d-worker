#!/usr/bin/env python3
"""
No patch needed - MAtCha's from_pretrained already handles local .pth files correctly.
This script is kept as a no-op for Dockerfile compatibility.
"""

def main():
    print("[PATCH] No patches needed - MAtCha handles local .pth files natively via load_model()")

if __name__ == "__main__":
    main()
