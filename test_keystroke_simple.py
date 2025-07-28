#!/usr/bin/env python3
"""
Simple keystroke test
"""
import os
import time

# Set display for headless environments
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

def test_keystrokes():
    print("Testing keystroke methods...")
    
    # Test 1: PyAutoGUI
    try:
        import pyautogui
        print("✅ PyAutoGUI imported successfully")
        pyautogui.press('x')
        print("✅ PyAutoGUI keystroke sent")
    except Exception as e:
        print(f"❌ PyAutoGUI failed: {e}")
    
    time.sleep(1)
    
    # Test 2: keyboard library
    try:
        import keyboard
        print("✅ keyboard library imported successfully")
        keyboard.press_and_release('z')
        print("✅ keyboard library keystroke sent")
    except Exception as e:
        print(f"❌ keyboard library failed: {e}")
    
    time.sleep(1)
    
    # Test 3: pynput
    try:
        from pynput.keyboard import Controller
        controller = Controller()
        print("✅ pynput imported successfully")
        controller.press('v')
        controller.release('v')
        print("✅ pynput keystroke sent")
    except Exception as e:
        print(f"❌ pynput failed: {e}")

if __name__ == "__main__":
    print("🧪 Simple Keystroke Test")
    print("=" * 30)
    test_keystrokes()
    print("\nTest complete. Check if any keys were actually pressed in your game/application.")