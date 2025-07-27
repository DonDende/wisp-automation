#!/usr/bin/env python3
"""
Wisp Video Analysis Helper
Calls Director's video analysis tools to extract complex data for wisp automation

This script serves as a bridge between the wisp automation folder and Director's tools.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_wisp_video(video_path: str, output_dir: str = None) -> bool:
    """
    Analyze wisp video using Director's advanced tools
    
    Args:
        video_path: Path to the wisp summoning video
        output_dir: Directory to save analysis results (default: current directory)
    
    Returns:
        bool: True if analysis succeeded, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"❌ Video file not found: {video_path}")
        return False
    
    # Set up paths
    current_dir = Path(__file__).parent
    director_dir = current_dir.parent / "Director For Videos"
    
    if not director_dir.exists():
        logger.error(f"❌ Director directory not found: {director_dir}")
        return False
    
    if output_dir is None:
        output_dir = current_dir / "analysis_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Prepare analysis command
    analysis_script = director_dir / "wisp_video_analyzer.py"
    output_file = output_dir / "wisp_analysis.json"
    templates_dir = output_dir / "automation_templates"
    
    if not analysis_script.exists():
        logger.error(f"❌ Analysis script not found: {analysis_script}")
        return False
    
    logger.info("🎬 Starting wisp video analysis using Director")
    logger.info(f"   Video: {video_path}")
    logger.info(f"   Output: {output_file}")
    logger.info(f"   Templates: {templates_dir}")
    
    try:
        # Build command to run Director analysis
        cmd = [
            "bash", "-c",
            f"cd '{director_dir}/backend' && source venv/bin/activate && cd .. && "
            f"python wisp_video_analyzer.py '{video_path}' "
            f"--output '{output_file}' --templates-dir '{templates_dir}' --verbose"
        ]
        
        logger.info("🔄 Running Director analysis...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=director_dir)
        
        if result.returncode == 0:
            logger.info("✅ Director analysis completed successfully")
            
            # Check if output files were created
            if output_file.exists():
                logger.info(f"📊 Analysis results saved: {output_file}")
                
                # Show summary of results
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"📈 Analysis data size: {len(json.dumps(data))} characters")
                    logger.info(f"📋 Analysis sections: {list(data.keys())}")
                except Exception as e:
                    logger.warning(f"Could not read analysis results: {e}")
            
            if templates_dir.exists():
                template_files = list(templates_dir.glob("*"))
                logger.info(f"📁 Generated {len(template_files)} template files")
                for template_file in template_files:
                    logger.info(f"   - {template_file.name}")
            
            return True
            
        else:
            logger.error("❌ Director analysis failed")
            logger.error(f"   Error output: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run Director analysis: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze wisp video using Director")
    parser.add_argument("video_path", help="Path to wisp summoning video")
    parser.add_argument("--output-dir", help="Directory to save analysis results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = analyze_wisp_video(args.video_path, args.output_dir)
    
    if success:
        logger.info("🎉 Wisp video analysis completed!")
        logger.info("   You can now use the generated configuration with:")
        logger.info("   python wisp_automation_enhanced.py --config analysis_results/automation_templates/wisp_automation_config.json")
    else:
        logger.error("❌ Wisp video analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()