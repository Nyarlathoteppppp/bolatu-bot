import asyncio
import time

from qq_social_agent.tools.fresh_context import FreshContextTool, FreshItem
from qq_social_agent.tools.safe_url_reader import UrlReadResult


def test_page_budget_preserves_finished_source_and_cancels_slow_one():
    tool=FreshContextTool(provider='tavily',tavily_api_key='test')
    items=(FreshItem('Python docs','python','',url='https://docs.python.org/fast'),
           FreshItem('Python reference','python','',url='https://docs.python.org/slow'))
    cancelled=[]
    async def read(url):
        if url.endswith('/fast'):
            return UrlReadResult('ok',url,text='Verified reference text',title='Python docs')
        try:
            await asyncio.sleep(3)
        finally:
            cancelled.append(url)
    tool._read_one_followup_url=read
    pages,_=asyncio.run(tool._read_followup_pages(items,query='Python',deadline=time.monotonic()+0.03))
    assert len(pages)==1
    assert pages[0].text=='Verified reference text'
    assert cancelled==['https://docs.python.org/slow']


def test_total_lookup_budget_includes_page_fetching():
    tool=FreshContextTool(provider='tavily',tavily_api_key='test',timeout_seconds=1,followup_search_hops=1)
    item=FreshItem('Reference','official','',summary='Useful reference',url='https://docs.python.org/slow')
    async def provider(*args,**kwargs):return '',(item,)
    async def read(url):await asyncio.sleep(4)
    tool._lookup_provider=provider
    tool._read_one_followup_url=read
    started=time.monotonic()
    result=asyncio.run(tool.lookup('Python API reference',kind='web'))
    assert time.monotonic()-started<1.5
    assert result.status=='ok'
    assert result.items
    assert not result.page_texts


def test_optional_usefulness_judge_cannot_exceed_deadline_or_discard_sources():
    tool=FreshContextTool(provider='tavily',tavily_api_key='test',timeout_seconds=1,
                          followup_page_max_tries=0,followup_search_hops=1)
    item=FreshItem('Reference','official','',summary='Useful reference',url='https://docs.python.org/')
    async def provider(*args,**kwargs):return '',(item,)
    async def judge(**kwargs):await asyncio.sleep(4)
    tool._lookup_provider=provider
    tool._judge_search_useful=judge
    started=time.monotonic()
    result=asyncio.run(tool.lookup('Python API reference',kind='web'))
    assert time.monotonic()-started<1.5
    assert result.status=='ok'
    assert result.items
