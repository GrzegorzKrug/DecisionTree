import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import warnings
import re


class RuleBase:
    def __init__(self):
        self.outcome = None
        self.next_step = None

        self.desc = ""
        self.short_desc = ""

        self.eq = None
        self.sm = None
        self.sme = None
        self.big = None
        self.bige = None
        self.diff = None

    @staticmethod
    def check_boolean(value):
        if value:
            return True
        else:
            return False

    def check_equal(self, value):
        if value == self.eq:
            return True
        else:
            return False

    def check_diff(self, value):
        if value != self.diff:
            return True
        else:
            return False

    def check_smaller(self, value):
        if value < self.sm:
            return True
        else:
            return False

    def check_smaller_equal(self, value):
        if value <= self.sme:
            return True
        else:
            return False

    def check_bigger(self, value):
        if value > self.big:
            return True
        else:
            return False

    def check_bigger_equal(self, value):
        if value >= self.bige:
            return True
        else:
            return False


class RuleClass(RuleBase):
    def __init__(self, eq=None, bl=None,
                 diff=None, sm=None, sme=None, big=None, bige=None,
                 next_step=None, outcome=None):
        super().__init__()

        self.conds = []

        self.setup_conditions(bl=bl, eq=eq, diff=diff,
                              sm=sm, sme=sme,
                              big=big, bige=bige)

        if next_step is not None and outcome is not None:
            raise ValueError(f"Outcome value and next step can not be both None!")

        elif next_step and outcome:
            raise ValueError(f"Outcome value and next step can not be used simultaneously")

        else:
            self.next_step = next_step
            self.outcome = outcome

    def setup_conditions(self, bl=None, eq=None, diff=None, sm=None, sme=None, big=None, bige=None):
        if bl:
            self.desc = "Boolean Check"
            self.short_desc = "True"
            conds = [self.check_boolean]

        elif sm and sme or big and bige or eq and diff:
            raise ValueError("Can not interpret conditions. Some types are not allowed simultaneously")

        elif sm or sme or big or bige or diff:
            conds = []
            types = []
            if sm:
                sm = int(sm)
                self.sm = sm
                types.append(f"<{sm}")
                conds.append(self.check_smaller)

            elif sme:
                sme = int(sme)
                self.sme = sme
                types.append(f"<={sme}")
                conds.append(self.check_smaller_equal)

            if big:
                big = int(big)
                self.big = big
                types.append(f">{big}")
                conds.append(self.check_bigger)
            elif bige:
                bige = int(bige)
                self.bige = bige
                types.append(f">={bige}")
                conds.append(self.check_bigger_equal)

            if diff:
                diff = int(diff)
                self.diff = diff
                types.append(f"!={diff}")
                conds.append(self.check_diff)

            max_val = sm or sme
            min_val = big or bige

            if max_val and min_val and min_val > max_val:
                raise ValueError(f"Conditions are impossible: {min_val} < {max_val}")

            self.short_desc = " & ".join(types)
            self.desc = "Comparison check: " + self.short_desc

        elif eq:
            eq = int(eq)
            self.eq = eq
            self.desc = f"Equal == {eq}"
            self.short_desc = f"=={eq}"
            conds = [self.check_equal]

        else:
            warnings.warn(f"No condition. Using boolean checking")
            self.desc = "Boolean Check"
            self.short_desc = "True"
            conds = [self.check_boolean]

        self.conds = conds

    def check_conditions(self, value):
        for fun in self.conds:
            valid = fun(value)
            if not valid:
                return False
        return True

    def __str__(self):
        return str(self.desc)


class DecisionTree:
    def __init__(self):
        self.tree = {}
        self.fail = {}
        self.ord_symbols = [r"<=", r">=", r"<", r">", "==", "!="]
        self.symbols = set(self.ord_symbols)

        self.root = None
        self.nodes = None
        self.nodes_set = None

    def add_rule(self, name, conditions=None, outcome=None, next_step=None):
        if outcome:
            try:
                outcome = int(outcome)
            except ValueError:
                print(f"This is not number: {outcome} in {name} - {conditions}")
                return False

        if conditions:
            tmp_conditions = conditions
            conditions = conditions.replace(" ", "")
            pattern = "|".join(self.ord_symbols) + r"|\d+"
            conditions = re.findall(pattern, conditions)

            if len(conditions) == 1:
                try:
                    num = int(conditions[0])
                except ValueError as err:
                    print(f"This is not a number: {conditions[0]}")
                    num = None
                    return False
                # ruld = self.get_ruld(eq=num)
                rulob = RuleClass(eq=num, next_step=next_step, outcome=outcome)
                self._update_tree_rules(name, rulob)

            elif len(conditions) > 1:
                if not self.validate_parse(conditions):
                    print("Not valid")

                good_syms = self.symbols.copy()
                kwargs = {}
                for sym, num in zip(conditions[0::2], conditions[1::2]):
                    "This loop checks if there is not condition conflicts and duplicated symbols"

                    if len(good_syms) <= 1:
                        print(f"To much rules or incorrect combination: {conditions}")
                        return None

                    if sym == '<' or sym == '<=':
                        if sym == "<=":
                            kw = "sme"
                        else:
                            kw = 'sm'
                        this_types = ["<", "<="]
                        incorect = ["=="]

                    elif sym == '>' or sym == '>=':
                        if sym == ">=":
                            kw = "bige"
                        else:
                            kw = 'big'
                        this_types = [">", ">="]
                        incorect = ["=="]

                    elif sym == '==' or sym == '!=':
                        if sym == "==":
                            kw = "eq"
                        else:
                            kw = 'diff'
                        this_types = ["==", "!="]
                        incorect = ["<", "<=", ">", ">="]
                    else:
                        raise ValueError(f"Symbol unknown: {sym}")

                    for s in incorect:
                        try:
                            good_syms.remove(s)
                        except KeyError:
                            pass
                    try:
                        good_syms.remove(this_types[0])
                        good_syms.remove(this_types[1])
                    except KeyError as err:
                        if "!=" in good_syms:
                            pass
                        else:
                            print(f"Invalid rule: {name} - {conditions}")
                            print(f"{err}")
                            return False

                    kwargs.update({kw: num})
                ruldob = RuleClass(next_step=next_step, outcome=outcome, **kwargs)
                self._update_tree_rules(name, ruldob)

            else:
                warnings.warn(f"Conditions are empty, using boolean condition: ''{tmp_conditions}'")
                self._add_bool(name, outcome, next_step)
        else:
            self._add_bool(name, outcome, next_step)

    def add_fail(self, name, outcome=None, next_step=None):
        if outcome and next_step:
            raise ValueError(f"Can not set value and next step as fail outcome for: {name}")
        elif name in self.fail.keys():
            warnings.warn(f"Overwriting fail outcome: {name}")
        self.fail.update({name: {'outcome': outcome, 'next': next_step}})

    def _add_bool(self, name, outcome, next_step):
        rulob = RuleClass(bl=True, outcome=outcome, next_step=next_step)
        self._update_tree_rules(name, rulob)

    # def get_ruld(self, bl=False, sm=False, sme=False, big=False, bige=False, eq=False, diff=False):
    #     """
    #     Smaller
    #     Smaller equal
    #     Bigger
    #     Bigger equal
    #     Equal
    #     Different
    #     """
    #     ruld = dict().fromkeys(['bl', 'sm', 'sme', 'big', 'bige', 'eq', 'diff', 'val', 'next'], False)

    def _update_tree_rules(self, name, new_rule):
        current = self.tree.get(name, [])
        current += [new_rule]
        self.tree.update({name: current})

    def build_tree(self):
        self.check_tree_keys()
        # self.check_condition_conflicts()
        valid, cmt, (root, nodes) = self.check_graph_sequence()

        if not valid:
            print(f"Not valid: {cmt}, there is cycle in tree")
        else:
            self.root = root
            self.nodes = nodes
            self.nodes_set = set(nodes)
            print(f"Tree is valid, build complete")

    def predict(self, **kwargs):
        # print(self.root)
        # print(self.nodes)
        next_name = self.root
        while next_name:
            print(f"Checking: {next_name}")
            cur_name = next_name
            next_name = None
            outcome = False

            try:
                value = int(kwargs[cur_name])
            except KeyError:
                print(f"Can not predict, value is missing for: {cur_name}")
                return None
            except ValueError:
                print(f"This is not value: {kwargs[cur_name]}")
                return None

            cur_node = self.tree.get(cur_name)
            valid = False

            for rulob in cur_node:
                next_name = rulob.next_step
                outcome = rulob.outcome
                valid = rulob.check_conditions(value)
                if valid:
                    break

            if valid:
                print(f"Valid, out:{outcome}, next:{next_name}")
                if outcome:
                    return outcome
                elif next_name:
                    pass
            else:
                fail_cond = self.fail.get(cur_name)
                print(f"Not apply, {fail_cond}")
                try:
                    outcome, next_name = fail_cond['outcome'], fail_cond['next']
                except TypeError:
                    warnings.warn(f"Fail conditions does not exists, returning None, in {cur_name}")
                    return None
                if outcome is not None:
                    return outcome

    def check_tree_keys(self):
        fkeys = set(self.fail.keys())
        tree_keys = set(self.tree.keys())

        for key, rul_list in self.tree.items():
            if key not in fkeys:
                warnings.warn(f"{key} has no fail outcome. use 'dt.add_fail()'")
            for rulob in rul_list:
                nx = rulob.next_step
                if nx and nx not in tree_keys:
                    raise ValueError(f"This next step does not exist: '{nx}' in '{key}' rules")

        for key, ruld in self.fail.items():
            nx = ruld['next']
            if nx and nx not in tree_keys:
                raise ValueError(f"This next step does not exist: '{nx}' in '{key}' rules")

    def _get_roots_with_nodes(self):
        nodes = {key: {'visited': 0, 'childs': set(), 'parents': set()}
                 for key in self.tree.keys()}
        roots = set(self.tree.keys())

        for name, rules in self.tree.items():
            "Removing non roots from roots, creating child, parent relations"
            for rulob in rules:
                _next = rulob.next_step
                if _next:
                    this_node = nodes.get(name)
                    childs = this_node.get('childs')
                    childs.add(_next)
                    this_node.update({'childs': childs})
                    nodes.update({name: this_node})

                    child_node = nodes.get(_next)
                    parents = child_node.get('parents')
                    parents.add(name)
                    child_node.update({'parents': parents})
                    nodes.update({_next: child_node})

                    if _next in roots:
                        roots.remove(_next)

            fail_outcome = self.fail.get(name, None)
            if fail_outcome:
                _next = fail_outcome.get("next", None)
                if _next:
                    this_node = nodes.get(name)
                    childs = this_node.get('childs')
                    childs.add(_next)
                    this_node.update({'childs': childs})
                    nodes.update({name: this_node})

                    child_node = nodes.get(_next)
                    parents = child_node.get('parents')
                    parents.add(name)
                    child_node.update({'parents': parents})
                    nodes.update({_next: child_node})

                    if _next in roots:
                        roots.remove(_next)

                    if _next in roots:
                        roots.remove(_next)
        return roots, nodes

    def check_graph_sequence(self):
        roots, nodes = self._get_roots_with_nodes()

        if len(roots) != 1:
            err = f"Tree must have only one root: {roots}"
            print(err)
            return False, err

        root = list(roots)[0]
        visited = {root}
        goto_nodes = nodes.get(root).get('childs')

        n = 0
        while goto_nodes and n < 10:
            n += 1
            next_nodes = set()
            for cur_name in goto_nodes:
                if cur_name in visited:
                    err = f"This node was visited before: {cur_name}"
                    print(err)
                    return False, err

                checking_node = nodes[cur_name]
                vis_num = checking_node.get('visited', 0) + 1
                checking_node.update({'visited': vis_num})

                if 1 >= len(checking_node.get('parents')):
                    "Adding parents, its only 1, no conflicts"
                    visited.add(cur_name)
                    for ch in checking_node.get('childs'):
                        next_nodes.add(ch)
                elif vis_num >= len(checking_node.get('parents')):
                    "Joining paths"
                    visited.add(cur_name)
                    for ch in checking_node.get('childs'):
                        next_nodes.add(ch)
                else:
                    "Ignore / wait to join"

            goto_nodes = next_nodes

        if len(visited) == len(self.tree):
            return True, "Its ok to be ok", (root, nodes)
        else:
            not_vis = [node for node in nodes if node not in visited]
            return False, f"Some nodes was not visited {not_vis}", (None, None)

    def _get_network_nodes(self, pos, graph):
        node_names = []
        node_x = []
        node_y = []
        outcome_names = []
        outcome_x = []
        outcome_y = []
        for node in graph.nodes():
            x, y = pos[node]
            if node in self.nodes_set:
                node_names.append(str(node))
                node_x.append(x)
                node_y.append(y)
            else:
                outcome_names.append(str(node))
                outcome_x.append(x)
                outcome_y.append(y)

        edge_x = []
        edge_y = []
        edge_center_x = []
        edge_center_y = []
        offset = 7

        for p0, p1 in graph.edges():
            x0, y0 = pos[p0]
            x1, y1 = pos[p1]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)

            if x0 < x1:
                edge_center_x.append(np.mean([x0, x1]) + offset)
            else:
                edge_center_x.append(np.mean([x0, x1]) - offset)
            edge_center_y.append(np.mean([y0, y1]))

            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        return node_x, node_y, node_names, \
               edge_x, edge_y, edge_center_x, edge_center_y, \
               outcome_x, outcome_y, outcome_names

    def draw_graph_plotly(self, font_size=30):
        """

        Returns:

        """
        G = nx.DiGraph()
        next_list = [self.root]
        conds = []
        while next_list and self.nodes:
            tmp_list = next_list
            next_list = []
            for cur_name in tmp_list:
                that_node = self.tree[cur_name]
                for rule in that_node:
                    # print(f"adding rule")
                    conds.append(rule.short_desc)
                    if rule.outcome is not None:
                        G.add_edge(cur_name, rule.outcome)
                    else:
                        G.add_edge(cur_name, rule.next_step)
                        next_list.append(rule.next_step)
                if cur_name in self.fail:
                    conds.append('else')
                    if self.fail[cur_name]['outcome'] is not None:
                        G.add_edge(cur_name, self.fail[cur_name]['outcome'])
                    else:
                        G.add_edge(cur_name, self.fail[cur_name]['next'])
                        next_list.append(self.fail[cur_name]['next'])

        pos = graphviz_layout(G, prog='dot')

        (node_x, node_y, node_names,
         edge_x, edge_y, edge_center_x, edge_center_y,
         outcome_x, outcome_y, outcome_names) = self._get_network_nodes(pos, G)

        node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_names,
                textposition="top center",
                textfont={'size': font_size, 'color': 'red', 'family': "arial"},
                marker=dict(
                        # showscale=True,
                        # colorscale options
                        # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                        # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                        # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                        colorscale='YlGnBu',
                        reversescale=True,
                        color=[],
                        size=10,
                        # colorbar=dict(
                        #         thickness=15,
                        #         title='Node Connections',
                        #         xanchor='left',
                        #         titleside='right'
                        # ),
                        line_width=2)
        )

        outcome_trace = go.Scatter(
                x=outcome_x, y=outcome_y,
                mode='markers+text',
                hoverinfo='text',
                text=outcome_names,
                textposition="bottom center",
                marker={"size": 20, "color": "#14F"},
                textfont={'size': font_size, 'color': '#14F', 'family': "arial"}
        )

        edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines')

        cond_trace = go.Scatter(
                x=edge_center_x, y=edge_center_y,
                mode='text', text=conds,
                textfont={'size': font_size, 'color': 'black', 'family': "arial"},
                textposition='top center')

        layout = go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                        text="Decision Graph",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        fig = go.Figure(data=[edge_trace, node_trace, cond_trace, outcome_trace], layout=layout)
        fig.write_image("tree.jpg", width=1600, height=900)
        # fig.show()

    def validate_parse(self, rules_to_check):
        size = len(rules_to_check)
        if size == 1:
            "Must be equal to value"
            try:
                int(rules_to_check[0])
                return True
            except ValueError as err:
                return False
        elif size == 0:
            print(f"pared rules are empty: {rules_to_check}")
            return False

        elif not size % 2:
            return True
        else:
            print("Rules are not even, len:", size, rules_to_check)


dt = DecisionTree()

dt.add_rule('want', next_step='worktime')
dt.add_fail('want', outcome=0)

dt.add_rule('worktime', "<2", outcome=2)
dt.add_rule('worktime', "<3", outcome=3)
dt.add_rule('worktime', "<5", outcome=5)
dt.add_rule('worktime', ">=5", next_step='member')
dt.add_fail('worktime', next_step='criminal')

# dt.add_rule('worktime', "<1", outcome=5)
# dt.add_rule('worktime', ">=1<10", next_step='criminal')
# dt.add_rule('worktime', ">=10", next_step='member')

dt.add_rule('criminal', outcome=1)
dt.add_fail('criminal', outcome=5)

dt.add_rule('member', '<5', outcome=15)
# dt.add_rule('member', '==8', outcome=15)
# dt.add_rule('member', '>=5', outcome=35)
# # dt.add_rule('member', '>=5', next_step='criminal')
dt.add_fail('member', outcome=12)

dt.build_tree()

pred = dt.predict(want=True, worktime=10, criminal=False, member=9)
print(f"Predicted: {pred}")

dt.draw_graph_plotly()
