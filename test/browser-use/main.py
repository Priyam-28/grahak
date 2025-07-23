from browser_use.llm import ChatGoogle
from browser_use import Agent
from dotenv import load_dotenv
import asyncio

load_dotenv()

llm = ChatGoogle(model='gemini-2.0-flash')

async def main():
    agent = Agent(
        task="Go to amazon.com and tell me the price of this product and add to my cart. Here is the link: https://www.amazon.in/gp/aw/d/B0CK2TQWQQ/?_encoding=UTF8&pd_rd_plhdr=t&aaxitk=62e63f622def36da528686cae041f7aa&hsa_cr_id=0&sr=1-1-e0fa1fdd-d857-4087-adda-5bd576b25987&ref_=sbx_be_s_sparkle_dlcd_ls_dpt&psc=1",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())
