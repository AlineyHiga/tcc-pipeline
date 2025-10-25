#!/usr/bin/env python3
"""Test script for the AutoFix pipeline with detailed error handling."""

import sys
import traceback
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    try:
        print("🚀 Starting AutoFix Pipeline Test")
        
        # Test imports
        print("📦 Testing imports...")
        from app.main import AutoFixPipeline
        print("✅ Imports successful")
        
        # Initialize pipeline
        print("🔧 Initializing pipeline...")
        pipeline = AutoFixPipeline()
        print("✅ Pipeline initialized")
        
        # Run pipeline
        print("▶️ Running pipeline...")
        result = pipeline.run()
        print("✅ Pipeline completed successfully")
        print(f"📊 Results: {result}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()