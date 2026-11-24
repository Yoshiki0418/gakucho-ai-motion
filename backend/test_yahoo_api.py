import asyncio

from app.agent.general_conversation.tools.location_tools import (
    get_travel_info,
    search_nearby_places,
)


async def main():
    print("=== Testing search_nearby_places ===")
    try:
        result = search_nearby_places(place="大学", category="ラーメン", radius=500)
        print("Result:", result)
    except Exception as e:
        print("Error:", e)

    print("\n=== Testing get_travel_info ===")
    try:
        result = get_travel_info(origin="大学", destination="金沢駅", mode="車")
        print("Result:", result)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
