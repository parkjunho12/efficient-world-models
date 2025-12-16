#!/usr/bin/env python3
"""
Collect training data from CARLA simulator.

This script connects to a running CARLA server and collects episodes
of driving data (images + actions) for world model training.
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
import json


def collect_carla_data(
    num_episodes: int,
    output_dir: Path,
    town: str = 'Town01',
    weather: str = 'ClearNoon',
    num_vehicles: int = 50,
    num_pedestrians: int = 30,
    episode_length: int = 100
):
    """
    Collect data from CARLA simulator.
    
    Args:
        num_episodes: Number of episodes to collect
        output_dir: Output directory
        town: CARLA town name
        weather: Weather preset
        num_vehicles: Number of vehicles to spawn
        num_pedestrians: Number of pedestrians to spawn
        episode_length: Number of frames per episode
    """
    try:
        import carla
    except ImportError:
        print("✗ Error: CARLA not installed")
        print("Install with: pip install carla")
        print("Or download from: https://carla.org/")
        return False
    
    print("\n" + "=" * 80)
    print("CARLA Data Collection")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Town: {town}")
    print(f"Weather: {weather}")
    print(f"Episode length: {episode_length} frames")
    print("=" * 80)
    
    # Connect to CARLA
    print("\nConnecting to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("✓ Connected to CARLA")
    except Exception as e:
        print(f"✗ Error connecting to CARLA: {e}")
        print("\nMake sure CARLA server is running:")
        print("  ./CarlaUE4.sh")
        return False
    
    # Load town and set weather
    if world.get_map().name.split('/')[-1] != town:
        print(f"Loading town: {town}")
        world = client.load_world(town)
    
    # Set weather
    weather_presets = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
        'WetNoon': carla.WeatherParameters.WetNoon,
        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
    }
    
    if weather in weather_presets:
        world.set_weather(weather_presets[weather])
        print(f"✓ Set weather: {weather}")
    
    # Create output directory
    episodes_dir = output_dir / 'episodes'
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect episodes
    for episode_idx in range(num_episodes):
        print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")
        
        try:
            # Spawn vehicle
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
            
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print("✓ Spawned vehicle")
            
            # Attach camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '256')
            camera_bp.set_attribute('image_size_y', '256')
            camera_bp.set_attribute('fov', '90')
            
            camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            
            # Setup camera callback
            images = []
            
            def camera_callback(image):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha channel
                images.append(array)
            
            camera.listen(camera_callback)
            
            # Enable autopilot
            vehicle.set_autopilot(True)
            
            # Collect data
            actions = []
            
            for frame_idx in range(episode_length):
                world.tick()
                
                # Get vehicle control
                control = vehicle.get_control()
                action = np.array([
                    control.steer,      # Steering
                    control.throttle,   # Throttle
                    control.brake,      # Brake
                    1.0 if control.gear > 0 else 0.0  # Forward gear
                ], dtype=np.float32)
                
                actions.append(action)
                
                time.sleep(0.05)  # 20 FPS
            
            # Save episode
            episode_dir = episodes_dir / f'episode_{episode_idx:04d}'
            episode_dir.mkdir(exist_ok=True)
            
            # Save images
            images_dir = episode_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            for i, img_array in enumerate(images):
                img = Image.fromarray(img_array)
                img.save(images_dir / f'{i:03d}.jpg')
            
            # Save actions
            actions = np.array(actions)
            np.save(episode_dir / 'actions.npy', actions)
            
            # Save metadata
            metadata = {
                'episode_idx': episode_idx,
                'town': town,
                'weather': weather,
                'num_frames': len(images),
                'vehicle_type': vehicle_bp.id
            }
            
            with open(episode_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved episode with {len(images)} frames")
            
            # Cleanup
            camera.stop()
            camera.destroy()
            vehicle.destroy()
            
        except Exception as e:
            print(f"✗ Error in episode {episode_idx}: {e}")
            continue
    
    # Save dataset info
    dataset_info = {
        'dataset': 'carla',
        'num_episodes': num_episodes,
        'town': town,
        'weather': weather,
        'episode_length': episode_length,
        'image_size': [256, 256]
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ Data collection complete!")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print("\nNext step:")
    print(f"  python scripts/prepare_dataset.py --dataset carla --data-root {output_dir}")
    print("=" * 80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Collect training data from CARLA simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 100 episodes in Town01
  python scripts/collect_carla_data.py --num-episodes 100 --output data/carla

  # Collect in different town with rain
  python scripts/collect_carla_data.py --num-episodes 50 --town Town03 --weather HardRainNoon

Note:
  Make sure CARLA server is running before executing this script:
    cd /path/to/CARLA
    ./CarlaUE4.sh
        """
    )
    
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of episodes to collect (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/carla',
        help='Output directory (default: data/carla)'
    )
    
    parser.add_argument(
        '--town',
        type=str,
        default='Town01',
        help='CARLA town (default: Town01)'
    )
    
    parser.add_argument(
        '--weather',
        type=str,
        default='ClearNoon',
        choices=['ClearNoon', 'CloudyNoon', 'WetNoon', 'SoftRainNoon', 'HardRainNoon'],
        help='Weather preset (default: ClearNoon)'
    )
    
    parser.add_argument(
        '--num-vehicles',
        type=int,
        default=50,
        help='Number of vehicles (default: 50)'
    )
    
    parser.add_argument(
        '--num-pedestrians',
        type=int,
        default=30,
        help='Number of pedestrians (default: 30)'
    )
    
    parser.add_argument(
        '--episode-length',
        type=int,
        default=100,
        help='Frames per episode (default: 100)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    success = collect_carla_data(
        num_episodes=args.num_episodes,
        output_dir=output_dir,
        town=args.town,
        weather=args.weather,
        num_vehicles=args.num_vehicles,
        num_pedestrians=args.num_pedestrians,
        episode_length=args.episode_length
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())