#!/usr/bin/env python3
"""Test MCP server for Apple Q4 2023 revenue."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_apple_financials():
    """Test getting Apple's financial data."""

    server_params = StdioServerParameters(
        command="/Applications/Docker.app/Contents/Resources/bin/docker",
        args=[
            "run", "-i", "--rm",
            "-e", "SEC_EDGAR_USER_AGENT=FinVet finvet@example.com",
            "sec-edgar-mcp:local"
        ]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("=" * 80)
            print("TEST: Apple Financials (Q4 2023 Revenue)")
            print("=" * 80)

            # Test 1: Get general financials
            print("\n1. Testing get_financials for Apple...")
            try:
                result = await session.call_tool(
                    "get_financials",
                    arguments={
                        "identifier": "AAPL",
                        "statement_type": "income"
                    }
                )

                print(f"\n✅ get_financials returned successfully!")
                print(f"Type: {type(result)}")
                print(f"Content: {result.content}")

                # Extract text from content
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"\nData:\n{item.text[:500]}...")  # First 500 chars

            except Exception as e:
                print(f"❌ Error: {e}")

            # Test 2: Get company facts
            print("\n\n2. Testing get_company_facts for Apple...")
            try:
                result = await session.call_tool(
                    "get_company_facts",
                    arguments={"identifier": "AAPL"}
                )

                print(f"\n✅ get_company_facts returned successfully!")
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"\nData:\n{item.text[:500]}...")

            except Exception as e:
                print(f"❌ Error: {e}")

            # Test 3: Get XBRL concepts for specific filing
            print("\n\n3. Testing get_xbrl_concepts for Apple FY2023 10-K...")
            try:
                result = await session.call_tool(
                    "get_xbrl_concepts",
                    arguments={
                        "identifier": "AAPL",
                        "accession_number": "0000320193-23-000106",  # FY2023 10-K
                        "concepts": ["Revenues", "RevenuesFromContractWithCustomer", "NetIncomeLoss"]
                    }
                )

                print(f"\n✅ get_xbrl_concepts returned successfully!")
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"\nData:\n{item.text}")

            except Exception as e:
                print(f"❌ Error: {e}")

            # Test 4: Compare periods (for Q4 derivation)
            print("\n\n4. Testing compare_periods (Annual vs 9-month for Q4)...")
            try:
                result = await session.call_tool(
                    "compare_periods",
                    arguments={
                        "identifier": "AAPL",
                        "metric": "Revenues",
                        "start_year": 2022,
                        "end_year": 2023
                    }
                )

                print(f"\n✅ compare_periods returned successfully!")
                for item in result.content:
                    if hasattr(item, 'text'):
                        print(f"\nData:\n{item.text}")

            except Exception as e:
                print(f"❌ Error: {e}")

            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_apple_financials())
