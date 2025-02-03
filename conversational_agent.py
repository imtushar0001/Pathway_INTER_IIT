import openai
from openai import OpenAI
import numpy as np


class ConversationalPipeline:

    def __init__(self, openai_api_key, model="gpt-4o"):
        openai.api_key = openai_api_key
        self.api_key = openai_api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)


    def call_openai(self, prompt, model="gpt-4"):
        # response = openai.ChatCompletion.create(
        #     model=model,
        #     messages=[{"role": "system", "content": "You are a helpful assistant."},
        #               {"role": "user", "content": prompt}]
        # )
        # return response['choices'][0]['message']['content']

        response = self.client.chat.completions.create(
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            model=self.model,
        )
        return response.choices[0].message.content

    def analyst_task(self, query, context):
        analyst_prompt = f"""
        You are an Analyst. Your task is to analyze the following query using the provided document context:

        Query: {query}
        Document Context: {context}

        Provide an informative response based on the query. If further clarification is needed, suggest a follow-up question. If no follow-up is needed, provide a conclusion.
        """
        return self.call_openai(analyst_prompt)

    def leader_task(self, response_1, response_2, query, context):
        leader_prompt = f"""
        You are the Leader. Your task is to unify and summarize the responses from Analyst 1 and Analyst 2 into a coherent final response, given the query and the context:

        Query: {query}
        Context: {context}

        Analyst 1 Response: {response_1}
        Analyst 2 Response: {response_2}

        Provide the unified response below.
        """
        return self.call_openai(leader_prompt)



    def check_follow_up(self, query, context, final_response):
        follow_up_prompt = f"""
        Based on the provided query, context, and final response, determine if the query has been fully answered.

        Query: {query}
        Context: {context}
        Final Response: {final_response}

        Output "Yes" if the query is fully answered, otherwise output "No."
        """
        return self.call_openai(follow_up_prompt).strip()

    def divide_correct_task_into_subtasks(self, query, context):

        divide_prompt = f"""
        The user has a query and a context. Your first task is to determine whether the query is a simple query or complex one i.e., determine whether there is a need of dividing the query into simple ones or not.
        If the query is simple, then keep the subtask1 as initial query and subtask 2 as empty, else your next task is to divide this query into two distinct subtasks that can be worked on independently.
        Subtask 1: Focuses on one fundamental aspect of the query.
        Subtask 2: Focuses on a complementary or orthogonal aspect to Subtask 1, without overlapping in scope or objectives.
        Ensure that the subtasks are distinct, independent, and non-redundant.
        Instructions:
        Analyze the provided query and context.
        Identify two subtasks that:
        Together contribute to solving the overall query.
        Individually focus on different aspects of the problem to avoid redundancy.
        Ensure the subtasks are well-defined and actionable, with clear objectives.
        Here is the query and context:
        Query: "{query}"
        Context: "{context}"

        Provide the output in this format:

        Subtask 1: [In case of simple query, keep the initial query here else in case of complex query keep the first independent subtask with clear and actionable instructions]
        Subtask 2: [In case of simple query, keep this empty else in case of complex query keep the second distinct subtask that complements the first]
        """
        response = self.call_openai(divide_prompt)
        subtask_1, subtask_2 = response.split("Subtask 1:")[1].split("Subtask 2:")
        return subtask_1.strip(), subtask_2.strip()

    def divide_incorrect_task_into_subtasks(self, query):

        divide_prompt = f"""
        The user has a query. Your first task is to determine whether the query is a simple query or complex one i.e., determine whether there is a need of dividing the query into simple ones or not.
        If the query is simple, then keep the subtask1 as initial query and subtask 2 as empty, else your next task is to divide this query into two distinct subtasks that can be worked on independently.
        Subtask 1: Focuses on one fundamental aspect of the query.
        Subtask 2: Focuses on a complementary or orthogonal aspect to Subtask 1, without overlapping in scope or objectives.
        Ensure that the subtasks are distinct, independent, and non-redundant.
        Instructions:
        Analyze the provided query.
        Identify two subtasks that:
        Together contribute to solving the overall query.
        Individually focus on different aspects of the problem to avoid redundancy.
        Ensure the subtasks are well-defined and actionable, with clear objectives.
        Here is the query:
        Query: "{query}"

        Provide the output in this format:

        Subtask 1: [In case of simple query, keep the initial query here else in case of complex query keep the first independent subtask with clear and actionable instructions]
        Subtask 2: [In case of simple query, keep this empty else in case of complex query keep the second distinct subtask that complements the first]
        """
        response = self.call_openai(divide_prompt)
        subtask_1, subtask_2 = response.split("Subtask 1:")[1].split("Subtask 2:")
        return subtask_1.strip(), subtask_2.strip()

    def generate_new_subtasks(self, query, subtask_1, subtask_2, context):
        generate_new_subtasks_prompt = f"""
        You are provided with a query, its context, and two previously defined subtasks (Subtask 1 and Subtask 2). Your task is to generate two new subtasks: Subtask 3 and Subtask 4, ensuring they are:

        Distinct from both Subtask 1 and Subtask 2.
        Independent of each other but contribute toward solving the original query.
        Complementary to the overall solution while addressing unexplored aspects of the query.
        Instructions:
        Analyze the provided query and context carefully.
        Avoid reusing or overlapping with the objectives of Subtask 1 and Subtask 2.
        Identify two new subtasks (Subtask 3 and Subtask 4) that focus on unexplored dimensions, technical considerations, or potential enhancements.
        Make each subtask clear, actionable, and non-redundant with previous subtasks.
        Here is the query, context, and previous subtasks:
        Query: "{query}"
        Context: "{context}"
        Subtask 1: {subtask_1}
        Subtask 2: {subtask_2}

        Provide the output in this format:

        Subtask 3: [Third independent subtask, distinct from Subtask 1 and Subtask 2, focusing on a new actionable aspect of the query]
        Subtask 4: [Fourth independent subtask, distinct from Subtask 1, Subtask 2, and Subtask 3, complementing the overall solution]
        """
        response = self.call_openai(generate_new_subtasks_prompt)
        subtask_3, subtask_4 = response.split("Subtask 3:")[1].split("Subtask 4:")
        return subtask_3.strip(), subtask_4.strip()

    def run_pipeline(self, query, context_a, context_b, subtask_1, subtask_2):

        response_1 = self.analyst_task(subtask_1, context_a)
        response_2 = self.analyst_task(subtask_2, context_b)

        final_response = self.leader_task(response_1, response_2, query, context_a + context_b)

        return final_response

    def final_unification_task(self, combined_response, response_3, response_4, query, context):
        unification_prompt = f"""
        You are the Leader. Your task is to unify and summarize all three analyst responses into a single, coherent, and comprehensive final response, given the query and the context:

        query: {query}
        context: {context}

        Analyst 1_2 Combined Response : {combined_response}
        Analyst 3 Response: {response_3}
        Analyst 4 Response: {response_4}

        Provide the unified response below, ensuring clarity, accuracy, and coherence.
        """
        return self.call_openai(unification_prompt)


    def run_pipeline_if_needed(self, query, context_c, context_d, subtask_3, subtask_4, final_response, context):


        response_3 = self.analyst_task(subtask_3, context_c)
        response_4 = self.analyst_task(subtask_4, context_d)

        context.extend(context_c)
        context.extend(context_d)


        final_follow_up_response = self.final_unification_task(final_response, response_3, response_4, query, context)
        # final_response = f"{final_response}\n\nFollow-up Response:\n{final_follow_up_response}"

        return final_response

